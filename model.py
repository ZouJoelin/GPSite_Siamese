import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter_mean, scatter_sum
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import unbatch
from data import *


class EdgeMLP(nn.Module):
    def __init__(self, num_hidden, dropout=0.2):
        super(EdgeMLP, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(num_hidden)
        self.W11 = nn.Linear(3*num_hidden, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V, edge_index, h_E):
        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)
        h_message = self.W12(self.act(self.W11(h_EV)))
        h_E = self.norm(h_E + self.dropout(h_message))
        return h_E


class Context(nn.Module):
    def __init__(self, num_hidden):
        super(Context, self).__init__()

        self.V_MLP_g = nn.Sequential(
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.Sigmoid()
                                )

    def forward(self, h_V, batch_id):
        c_V = scatter_mean(h_V, batch_id, dim=0)
        temp = c_V[batch_id]
        h_V = h_V * self.V_MLP_g(c_V[batch_id])
        return h_V
    

class GNNLayer(nn.Module):
    def __init__(self, num_hidden, dropout=0.2, num_heads=4):
        super(GNNLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(num_hidden) for _ in range(2)])

        self.attention = TransformerConv(in_channels=num_hidden, out_channels=int(num_hidden / num_heads), heads=num_heads, dropout = dropout, edge_dim = num_hidden, root_weight=False)
        self.PositionWiseFeedForward = nn.Sequential(
            nn.Linear(num_hidden, num_hidden*4),
            nn.ReLU(),
            nn.Linear(num_hidden*4, num_hidden)
        )
        self.edge_update = EdgeMLP(num_hidden, dropout)
        self.context = Context(num_hidden)

    def forward(self, h_V, edge_index, h_E, batch_id):
        dh = self.attention(h_V, edge_index, h_E)
        h_V = self.norm[0](h_V + self.dropout(dh))

        # Position-wise feedforward
        dh = self.PositionWiseFeedForward(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        # update edge
        h_E = self.edge_update(h_V, edge_index, h_E)

        # context node update
        h_V = self.context(h_V, batch_id)

        return h_V, h_E


class Graph_encoder(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, num_layers=4, drop_rate=0.2):
        super(Graph_encoder, self).__init__()
        
        self.node_embedding = nn.Linear(node_in_dim, hidden_dim, bias=True)
        self.edge_embedding = nn.Linear(edge_in_dim, hidden_dim, bias=True)
        self.norm_nodes = nn.BatchNorm1d(hidden_dim)
        self.norm_edges = nn.BatchNorm1d(hidden_dim)
        
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_e = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.layers = nn.ModuleList(
                GNNLayer(num_hidden=hidden_dim, dropout=drop_rate, num_heads=4)
            for _ in range(num_layers))

    def forward(self, h_V, edge_index, h_E, batch_id):
        h_V = self.W_v(self.norm_nodes(self.node_embedding(h_V)))
        h_E = self.W_e(self.norm_edges(self.edge_embedding(h_E)))

        for layer in self.layers:
            h_V, h_E = layer(h_V, edge_index, h_E, batch_id)
        
        return h_V


class SiameseGPSite(nn.Module): # Geometry-aware Protein Sequence-based predictor
    def __init__(self, node_input_dim=1217, edge_input_dim=450, hidden_dim=128, num_layers=4, dropout=0.2, augment_eps=0.1, 
                 group="train", d_embedding_path=None):
        """
        Args:
            - node_input_dim: 1024+9+184
            - edge_input_dim: 450
            - hidden_dim: 128
            - num_layers: 4
            - dropout: 0.2
            - augment_eps: 0.1
            - group: "train" or "valid" or "test"
            - d_embedding_path: path to save the d_embedding for further analysis
        """
        super(SiameseGPSite, self).__init__()
        self.group = group
        self.d_embedding_path = d_embedding_path
        self.augment_eps = augment_eps
        self.Graph_encoder = Graph_encoder(node_in_dim=node_input_dim, edge_in_dim=edge_input_dim, hidden_dim=hidden_dim, num_layers=num_layers, drop_rate=dropout)
        # self.norm_1 = nn.BatchNorm1d(hidden_dim)  # over num_residue for each dim

        self.KeepShapeMLP = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.projecter1D = nn.Linear(hidden_dim, 1)
        # self.norm_2 = nn.BatchNorm1d(hidden_dim)  # over num_residue for each dim

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_once(self, coord, h_V, edge_index, batch_id):
        # Data augmentation
        if self.training and self.augment_eps > 0:  # self.training is turn on when call model.train()
            coord = coord + self.augment_eps * torch.randn_like(coord)
            h_V = h_V + self.augment_eps * torch.randn_like(h_V)

        h_V_geo, h_E = get_geo_feat(coord, edge_index)
        h_V = torch.cat([h_V, h_V_geo], dim=-1)

        h_V = self.Graph_encoder(h_V, edge_index, h_E, batch_id) # [num_residue, hidden_dim]

        return h_V
    
    def forward(self, wt_graph: torch_geometric.data.Data, mut_graph: torch_geometric.data.Data):
        wt_embedding = self.forward_once(wt_graph.coord, wt_graph.node_feat, wt_graph.edge_index, wt_graph.batch)
        mut_embedding = self.forward_once(mut_graph.coord, mut_graph.node_feat, mut_graph.edge_index, mut_graph.batch)
        # both of shape: [num_residue, hidden_dim]
        # wt_embedding = self.norm_1(wt_embedding)
        # mut_embedding = self.norm_1(mut_embedding)

        wt_embedding = self.KeepShapeMLP(wt_embedding)
        mut_embedding = self.KeepShapeMLP(mut_embedding)
        assert (wt_graph.batch == mut_graph.batch).all()
        d_embedding = mut_embedding - wt_embedding
        # shape: [num_residue, hidden_dim]

        # save the d_embedding for further analysis while testing
        if self.group == "test" and self.d_embedding_path is not None:  
            for pair_name, d_emb in zip(mut_graph.name, unbatch(d_embedding, mut_graph.batch, dim=0)):
                torch.save(d_emb.cpu(), f"{self.d_embedding_path}/{pair_name}.pt")

        # d_embedding = self.norm_2(d_embedding)
        d_embedding = self.projecter1D(d_embedding)
        # shape: [num_residue, 1]
        # output = scatter_mean(d_embedding, wt_graph.batch, dim=0)
        output = scatter_sum(d_embedding, wt_graph.batch, dim=0)  # sum is more reasonable

        return output.squeeze()



##############  Geometric Featurizer  ##############
### checked!
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






def get_model(group="train", d_embedding_path=None):

    model = SiameseGPSite(group=group, d_embedding_path=d_embedding_path)
    return model



