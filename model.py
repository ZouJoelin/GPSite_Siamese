import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter_mean, scatter_sum
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import unbatch
from data import *
from geo import _positional_embeddings, _edge_feat_distance, _edge_feat_direction_orientation


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
    def __init__(self, node_input_dim=1217, edge_input_dim=450, hidden_dim=128, num_layers=4, dropout=0.2, 
                 group="train", d_embedding_path=None):
        """
        Args:
            - node_input_dim: 1024+9+184
            - edge_input_dim: 450
            - hidden_dim: 128
            - num_layers: 4
            - dropout: 0.2
            - group: "train" or "valid" or "test"
            - d_embedding_path: path to save the d_embedding for further analysis
        """
        super(SiameseGPSite, self).__init__()
        self.group = group
        self.d_embedding_path = d_embedding_path
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

    def forward_once(self, local_coord, coord, h_V, edge_index, batch_id):
        h_E = _get_geo_edge_feat(local_coord, coord, edge_index)

        h_V = self.Graph_encoder(h_V, edge_index, h_E, batch_id) # [num_residue, hidden_dim]

        return h_V
    
    def forward(self, wt_graph: torch_geometric.data.Data, mut_graph: torch_geometric.data.Data):
        wt_embedding = self.forward_once(wt_graph.local_coord, wt_graph.coord, wt_graph.node_feat, wt_graph.edge_index, wt_graph.batch)
        mut_embedding = self.forward_once(mut_graph.local_coord, mut_graph.coord, mut_graph.node_feat, mut_graph.edge_index, mut_graph.batch)
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

        return output.squeeze(-1)


##############  Geometric Featurizer  ##############
def _get_geo_edge_feat(local_coord, X, edge_index):
    # local_coord: [L, 3, 3] X: [L, 5, 3] edge_index: [2, 3235]

    edge_pos = _positional_embeddings(edge_index)  # [E, 16]
    edge_dist = _edge_feat_distance(X, edge_index)  # [E, 25 * 16]
    edge_direction, edge_orientation = _edge_feat_direction_orientation(X, edge_index, local_coord)  # [E, 2 * 5 * 3], [E, 4]
    geo_edge_feat = torch.cat([edge_pos, edge_orientation, edge_dist, edge_direction], dim=-1)

    return geo_edge_feat



def get_model(group="train", d_embedding_path=None):

    model = SiameseGPSite(group=group, d_embedding_path=d_embedding_path)
    return model



