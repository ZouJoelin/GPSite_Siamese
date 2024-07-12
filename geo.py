import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F


def get_geo_feat(X, edge_index):
    # X: [L, 5, 3] edge_index: [2, 3235]

    local_coord = _local_coord(X)  # [L, 3, 3]

    node_angles = _node_feat_angle(X)  # [L, 12]
    node_dist = _node_feat_distance(X)  # [L, 10 * 16]
    node_direction = _node_feat_direction(X, local_coord)  # [L, 4 * 3]

    edge_pos = _positional_embeddings(edge_index)  # [E, 16]
    edge_dist = _edge_feat_distance(X, edge_index)  # [E, 25 * 16]
    edge_direction, edge_orientation = _edge_feat_direction_orientation(X, edge_index, local_coord)  # [E, 2 * 5 * 3], [E, 4]
    
    geo_node_feat = torch.cat([node_angles, node_dist, node_direction], dim=-1)
    geo_edge_feat = torch.cat([edge_pos, edge_orientation, edge_dist, edge_direction], dim=-1)

    return geo_node_feat, geo_edge_feat


def _rbf(D, D_min=0., D_max=20., D_count=16):
    '''
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)

    return RBF


def _quaternions(R):
    """ Convert a batch of 3D rotations [R] to quaternions [Q]
        R [E,3,3]
        Q [E,4]
    """
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    Rxx, Ryy, Rzz = diag.unbind(-1)
    magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
          Rxx - Ryy - Rzz,
        - Rxx + Ryy - Rzz,
        - Rxx - Ryy + Rzz
    ], -1)))
    _R = lambda i,j: R[:,i,j]
    signs = torch.sign(torch.stack([
        _R(2,1) - _R(1,2),
        _R(0,2) - _R(2,0),
        _R(1,0) - _R(0,1)
    ], -1))
    xyz = signs * magnitudes
    # The relu enforces a non-negative trace
    w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
    Q = torch.cat((xyz, w), -1)
    Q = F.normalize(Q, dim=-1)

    return Q


def _local_coord(X):
    X_N = X[:,0]  # [L, 3]
    X_Ca = X[:,1]
    X_C = X[:,2]
    u = F.normalize(X_Ca - X_N, dim=-1)
    v = F.normalize(X_C - X_Ca, dim=-1)
    b = F.normalize(u - v, dim=-1)
    n = F.normalize(torch.cross(u, v), dim=-1)
    local_frame = torch.stack([b, n, torch.cross(b, n)], dim=-1) # [L, 3, 3] (3 column vectors)

    return local_frame  # [L, 3, 3]


def _node_feat_angle(X, eps=1e-7):
    # X: [L, 5, 3]
    # psi, omega, phi
    X = torch.reshape(X[:, :3], [3*X.shape[0], 3])  # [3*L, 3] (select and chain N, CA, C)
    dX = X[1:] - X[:-1]
    U = F.normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)  # D (+0, +pi) & (-0, -pi)
    D = F.pad(D, [1, 2]) # This scheme will remove phi[0], psi[-1], omega[-1]
    D = torch.reshape(D, [-1, 3])  # res_rotate: (phi, psi, omega)
    dihedral = torch.cat([torch.cos(D), torch.sin(D)], 1)

    # alpha, beta, gamma
    cosD = (u_2 * u_1).sum(-1) # alpha_{i}, gamma_{i}, beta_{i+1}
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.acos(cosD)
    D = F.pad(D, [1, 2])
    D = torch.reshape(D, [-1, 3])  # res_angle: (alpha, beta, gamma)
    bond_angles = torch.cat((torch.cos(D), torch.sin(D)), 1)

    node_angles = torch.cat((dihedral, bond_angles), 1)
    return node_angles  # [L, 12]


def _node_feat_distance(X):
    atom_N = X[:,0]  # [L, 3]
    atom_Ca = X[:,1]
    atom_C = X[:,2]
    atom_O = X[:,3]
    atom_R = X[:,4]

    node_list = ['Ca-N', 'Ca-C', 'Ca-O', 'N-C', 'N-O', 'O-C', 'R-N', 'R-Ca', "R-C", 'R-O']
    node_dist = []
    for pair in node_list:
        atom1, atom2 = pair.split('-')
        E_vectors = vars()['atom_' + atom1] - vars()['atom_' + atom2]
        rbf = _rbf(E_vectors.norm(dim=-1))
        node_dist.append(rbf)
    node_dist = torch.cat(node_dist, dim=-1) # dim = [N, 10 * 16]

    return node_dist  # [L, 10 * 16]


def _node_feat_direction(X, local_coord): # N, CA, C, O, R
    X_N = X[:,0]  # [L, 3]
    X_Ca = X[:,1]
    X_C = X[:,2]

    t = F.normalize(X[:, [0,2,3,4]] - X_Ca.unsqueeze(1), dim=-1) # [L, 4, 3]
    node_direction = torch.matmul(t, local_coord).reshape(t.shape[0], -1) # [L, 4 * 3]

    return node_direction  # [L, 4 * 3]


# Edge_features
def _positional_embeddings(edge_index, num_embeddings=16):
    # edge_index: [2, edge_num]
    d = edge_index[0] - edge_index[1]  # [edge_num]

    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=edge_index.device)
        * -(np.log(10000.0) / num_embeddings)
    )  # [8]
    angles = d.unsqueeze(-1) * frequency  # [edge_num, 1] * [8] (broadcast to [1, 8])
    # [edge_num, 8]
    PE = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    
    return PE  # [E, 16]


def _edge_feat_distance(X, edge_index):
    atom_N = X[:,0]  # [L, 3]
    atom_Ca = X[:,1]
    atom_C = X[:,2]
    atom_O = X[:,3]
    atom_R = X[:,4]

    atom_list = ["N", "Ca", "C", "O", "R"]
    edge_dist = []
    for atom1 in atom_list:
        for atom2 in atom_list:
            E_vectors = vars()['atom_' + atom1][edge_index[0]] - vars()['atom_' + atom2][edge_index[1]]
            rbf = _rbf(E_vectors.norm(dim=-1))
            edge_dist.append(rbf)
    edge_dist = torch.cat(edge_dist, dim=-1) # dim = [E, 25 * 16]

    return edge_dist  # [E, 25 * 16]


def _edge_feat_direction_orientation(X, edge_index, local_coord):
    X_N = X[:,0]  # [L, 3]
    X_Ca = X[:,1]
    X_C = X[:,2]

    node_j, node_i = edge_index

    t = F.normalize(X[node_j] - X_Ca[node_i].unsqueeze(1), dim=-1) # [E, 5, 3]
    edge_direction_ji = torch.matmul(t, local_coord[node_i]).reshape(t.shape[0], -1) # [E, 5 * 3]
    t = F.normalize(X[node_i] - X_Ca[node_j].unsqueeze(1), dim=-1) # [E, 5, 3]
    edge_direction_ij = torch.matmul(t, local_coord[node_j]).reshape(t.shape[0], -1) # [E, 5 * 3] # slightly improve performance
    edge_direction = torch.cat([edge_direction_ji, edge_direction_ij], dim = -1) # [E, 2 * 5 * 3]

    r = torch.matmul(local_coord[node_i].transpose(-1,-2), local_coord[node_j]) # [E, 3, 3]
    edge_orientation = _quaternions(r) # [E, 4]

    return edge_direction, edge_orientation  # [E, 2 * 5 * 3], [E, 4]


