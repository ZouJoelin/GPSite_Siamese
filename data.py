import numpy as np
import pandas as pd
import pickle as pkl

import torch
import torch.utils.data as data
import torch_geometric
from geo import _local_coord, _node_feat_angle, _node_feat_distance, _node_feat_direction


class Complex(object):
    def __init__(self, name: str, chains: list, seq, coord, node_feat,
                 jitter=None):
        """
        Attributes:
            name: str, name of the complex
            chains: list, list of chain ids
            chain_id: dict, {chain: torch.tensor}
            seq: dict, {chain: str}
            coord: dict, {chain: torch.tensor}
            node_feat: dict, {chain: torch.tensor}
            local_coord: dict, {chain: torch.tensor}
            geo_node_feat: dict, {chain: torch.tensor}
            """
        self.name = name
        self.chains = chains
        self.seq = seq
        self.coord = coord
        self.node_feat = node_feat
        self.chain_id = {chain: torch.full((len(seq[chain]),), i) for i, chain in enumerate(chains)}
        self.local_coord = None
        self.geo_node_feat = None

        if jitter:
            self._jitter(jitter)
        self.local_coord, self.geo_node_feat = self._get_geo_node_feat()
        self._concat()

    def _jitter(self, jitter=0.1):
        # _jitter must before _concat and _get_geo_node_feat
        assert isinstance(self.coord, dict) and self.geo_node_feat == None

        for chain in self.chains:
            self.coord[chain] = self.coord[chain] + jitter * torch.rand_like(self.coord[chain])
            self.node_feat[chain] = self.node_feat[chain] + jitter * torch.randn_like(self.node_feat[chain])

    def _get_geo_node_feat(self):
        # _get_geo_node_feat must before _concat
        assert isinstance(self.coord, dict)

        local_coords = {}
        geo_node_feats = {}
        for chain in self.chains:
            X = self.coord[chain]
            local_coord = _local_coord(X)  # [L, 3, 3]
            node_angles = _node_feat_angle(X)  # [L, 12]
            node_dist = _node_feat_distance(X)  # [L, 10 * 16]
            node_direction = _node_feat_direction(X, local_coord)  # [L, 4 * 3]
            geo_node_feat = torch.cat([node_angles, node_dist, node_direction], dim=-1)

            local_coords[chain] = local_coord
            geo_node_feats[chain] = geo_node_feat

        return local_coords, geo_node_feats

    def _concat(self):
        """
        seq: str -> str
        coord: dict -> torch.tensor
        norde_feat: dict -> torch.tensor
        chain_id: dict -> torch.tensor
        local_coord: dict -> torch.tensor
        geo_node_feat: dict -> torch.tensor
        """
        self.seq = "".join([self.seq[chain] for chain in self.chains])
        self.coord = torch.cat([self.coord[chain] for chain in self.chains])
        self.node_feat = torch.cat([self.node_feat[chain] for chain in self.chains])
        self.chain_id = torch.cat([self.chain_id[chain] for chain in self.chains])

        self.local_coord = torch.cat([self.local_coord[chain] for chain in self.chains])
        self.geo_node_feat = torch.cat([self.geo_node_feat[chain] for chain in self.chains])


class SiameseProteinGraphDataset(data.Dataset):
    def __init__(self, data: pd.DataFrame, feature_path="./data/", graph_mode="radius", radius=15, top_k=30,
                 training=False, augment_eps=0.1) -> None:
        super(SiameseProteinGraphDataset, self).__init__()

        self.data = data
        self.feature_path = feature_path
        self.graph_mode = graph_mode
        if self.graph_mode == "radius":
            self.radius = radius
        elif self.graph_mode == "knn":
            self.top_k = top_k
        self.training = training
        self.augment_eps = augment_eps if self.training else None
        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                'N': 2, 'Y': 18, 'M': 12}

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entry = self.data.iloc[idx]
        wt_name = entry['wt_name']
        mut_name = entry['mut_name']
        # if len(wt_name.split('_')) == 3 or len(wt_name.split('_')) == 1:  # wt_name: 1A4Y_A_B/1A4Y; mut_name: 1A4Y_A_B_DA435A
        #     wt_name = wt_name.split('_')[0]
        # else:                             # wt_name: 1A4Y_A_B_DA435A; mut_name: 1A4Y_A_B
        #     mut_name = mut_name.split('_')[0]
        y = entry['target']  # numpy.float64
        # print(f"wt_name: {wt_name}; mut_name: {mut_name}")
        
        wt_complex = self.get_complex(wt_name, jitter=self.augment_eps)
        mut_complex = self.get_complex(mut_name, jitter=self.augment_eps)

        wt_graph = self.build_graph(wt_complex)
        mut_graph = self.build_graph(mut_complex)

        # wt_graph = self.featurize_graph(wt_name)
        # mut_graph = self.featurize_graph(mut_name)

        y = torch.tensor(y, dtype=torch.float32)
        return mut_name, wt_graph, mut_graph, y
    
    def get_complex(self, name, jitter):
        with open(f"{self.feature_path}/seq/{name}.pkl", 'rb') as seq_file:
            seq = pkl.load(seq_file)
        chains = list(seq.keys())
        coord = torch.load(f"{self.feature_path}/coord/{name}.pt")
        ProtTrans_feature = torch.load(f"{self.feature_path}/ProtTrans/{name}.pt")
        DSSP_feature = torch.load(f"{self.feature_path}/DSSP/{name}.pt")
        pre_computed_node_feature = {chain: torch.cat([ProtTrans_feature[chain], DSSP_feature[chain]], dim=-1).to(torch.float32) for chain in chains}

        complex = Complex(name, chains, seq, coord, pre_computed_node_feature, jitter)
        return complex

    def build_graph(self, complex: Complex):
        with torch.no_grad():
            X_ca = complex.coord[:, 1]
            # radius_graph -> knn_graph: less memory requirement
            if self.graph_mode == "radius":
                edge_index = torch_geometric.nn.radius_graph(X_ca, r=self.radius, loop=True, max_num_neighbors=1000, num_workers=4)
            elif self.graph_mode == "knn":
                edge_index = torch_geometric.nn.knn_graph(X_ca, k=self.top_k)
            # edge_index.shape: [2, edges_num]
        
        node_feat = torch.cat([complex.node_feat, complex.geo_node_feat], dim=-1)
        graph_data = torch_geometric.data.Data(name=complex.name, seq=complex.seq, coord=complex.coord, local_coord=complex.local_coord,
                                               node_feat=node_feat, edge_index=edge_index)
        return graph_data
    

    def featurize_graph(self, name):
        with torch.no_grad():
            with open(f"{self.feature_path}/seq/{name}.txt") as seq_file:
                seq = seq_file.readline()
            coord = torch.load(f"{self.feature_path}/coord/{name}.pt")
            ProtTrans_feature = torch.load(f"{self.feature_path}/ProtTrans/{name}.pt")
            DSSP_feature = torch.load(f"{self.feature_path}/DSSP/{name}.pt")

            pre_computed_node_feature = torch.cat([ProtTrans_feature, DSSP_feature], dim=-1)
            X_ca = coord[:, 1]
            # radius_graph -> knn_graph: less memory requirement
            if self.graph_mode == "radius":
                edge_index = torch_geometric.nn.radius_graph(X_ca, r=self.radius, loop=True, max_num_neighbors=1000, num_workers=4)
            elif self.graph_mode == "knn":
                edge_index = torch_geometric.nn.knn_graph(X_ca, k=self.top_k)  
            # edge_index.shape: [2, edges_num]
        graph_data = torch_geometric.data.Data(name=name, seq=seq, coord=coord, node_feat=pre_computed_node_feature, edge_index=edge_index)
        return graph_data


class SiameseProteinGraphDataset_prediction(data.Dataset):
    def __init__(self, data: pd.DataFrame, feature_path="./data/", graph_mode="radius", radius=15, top_k=30) -> None:
        super(SiameseProteinGraphDataset_prediction, self).__init__()

        self.data = data
        self.feature_path = feature_path
        self.graph_mode = graph_mode
        if self.graph_mode == "radius":
            self.radius = radius
        elif self.graph_mode == "knn":
            self.top_k = top_k
        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                'N': 2, 'Y': 18, 'M': 12}

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entry = self.data.iloc[idx]
        wt_name = entry['wt_name']
        mut_name = entry['mut_name']
        # if len(wt_name.split('_')) == 3 or len(wt_name.split('_')) == 1:  # wt_name: 1A4Y_A_B/1A4Y; mut_name: 1A4Y_A_B_DA435A
        #     wt_name = wt_name.split('_')[0]
        # else:                             # wt_name: 1A4Y_A_B_DA435A; mut_name: 1A4Y_A_B
        #     mut_name = mut_name.split('_')[0]
        # print(f"wt_name: {wt_name}; mut_name: {mut_name}")
        
        wt_complex = self.get_complex(wt_name)
        mut_complex = self.get_complex(mut_name)

        wt_complex._concat()
        mut_complex._concat()

        wt_graph = self.build_graph(wt_complex)
        mut_graph = self.build_graph(mut_complex)

        return mut_name, wt_graph, mut_graph

    def get_complex(self, name):
        with open(f"{self.feature_path}/seq/{name}.pkl", 'rb') as seq_file:
            seq = pkl.load(seq_file)
        chains = list(seq.keys())
        coord = torch.load(f"{self.feature_path}/coord/{name}.pt")
        ProtTrans_feature = torch.load(f"{self.feature_path}/ProtTrans/{name}.pt")
        DSSP_feature = torch.load(f"{self.feature_path}/DSSP/{name}.pt")
        pre_computed_node_feature = {chain: torch.cat([ProtTrans_feature[chain], DSSP_feature[chain]], dim=-1).to(torch.float32) for chain in chains}

        complex = Complex(name, chains, seq, coord, pre_computed_node_feature)
        return complex

    def build_graph(self, complex):
        with torch.no_grad():
            X_ca = complex.coord[:, 1]
            # radius_graph -> knn_graph: less memory requirement
            if self.graph_mode == "radius":
                edge_index = torch_geometric.nn.radius_graph(X_ca, r=self.radius, loop=True, max_num_neighbors=1000, num_workers=4)
            elif self.graph_mode == "knn":
                edge_index = torch_geometric.nn.knn_graph(X_ca, k=self.top_k)
            # edge_index.shape: [2, edges_num]
        graph_data = torch_geometric.data.Data(name=complex.name, seq=complex.seq, coord=complex.coord, node_feat=complex.node_feat, edge_index=edge_index)
        return graph_data


def get_data(dataset_path = "./data/dataset_processed.pt") -> pd.DataFrame:
    # dataset_train = torch.load(f"{data_path}/dataset_train.pt")
    # dataset_test = torch.load(f"{data_path}/dataset_test.pt")

    # dataset_train = SiameseProteinGraphDataset(dataset_train, feature_path="./data/", radius=15)
    # dataset_test = SiameseProteinGraphDataset(dataset_test, feature_path="./data/", radius=15)
    dataset = torch.load(dataset_path)

    return dataset











