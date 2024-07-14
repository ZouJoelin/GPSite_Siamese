import numpy as np
import pandas as pd
import pickle as pkl


import torch
import torch.utils.data as data
import torch_geometric
from sklearn.metrics import pairwise_distances
from Bio.PDB import PDBParser

from geo import _local_coord, _node_feat_angle, _node_feat_distance, _node_feat_direction


class Complex(object):
    def __init__(self, name: str, chains, seq, coord, node_feat,
                 jitter=None):
        """
        Attributes:
            name: str, name of the complex
            chains: dict, {chain: int}
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
        self.chain_id = {chain: torch.full((len(seq[chain]),), i) for chain, i in chains.items()}
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

    def _mask_select(self, mask):
        # _mask_select must after _concat
        assert isinstance(self.seq, str)
        mask = mask.astype('bool')
        assert len(mask) == len(self.seq)

        self.seq = ''.join([res for res, m in zip(self.seq, mask) if m])
        self.chain_id = self.chain_id[mask]
        self.coord = self.coord[mask]
        self.local_coord = self.local_coord[mask]
        self.node_feat =self.node_feat[mask]
        self.geo_node_feat = self.geo_node_feat[mask]


class SiameseProteinGraphDataset(data.Dataset):
    def __init__(self, data: pd.DataFrame, feature_path="./data/", graph_mode="radius", radius=15, top_k=30,
                 training=False, augment_eps=0.1, cut_interface=True, cut_mutation=True, contact_threshold=10) -> None:
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
        self.cut_interface = cut_interface
        self.cut_mutation = cut_mutation
        self.contact_threshold = contact_threshold
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

        y = entry['target']  # numpy.float64
        # print(f"wt_name: {wt_name}; mut_name: {mut_name}")
        
        wt_complex = self.get_complex(wt_name, jitter=self.augment_eps)
        mut_complex = self.get_complex(mut_name, jitter=self.augment_eps)

        mut_name = mut_name if len(mut_name.split('_')) > 1 else wt_name
        chains = wt_complex.chains
        chain_id = wt_complex.chain_id
        
        # cut interface & mutation surrounding
        if self.cut_interface or self.cut_mutation:
            mask = np.zeros(len(wt_complex.seq)).astype('bool')

            # get interface_index
            if self.cut_interface:
                coord = wt_complex.coord
                ca_wt = coord[:, 1]
                pair_dis_matrix = pairwise_distances(ca_wt, ca_wt)

                contact_dis_matrix = pair_dis_matrix.copy()
                for chain, i in chains.items():
                    isin = np.isin(chain_id, i)
                    self_chain_mask = np.multiply.outer(isin, isin)
                    contact_dis_matrix[self_chain_mask] = 10000
                interface_index = list(set(np.where(contact_dis_matrix < self.contact_threshold)[0]))
                mask[interface_index] = True

            # get mutation_surrounding_index
            if self.cut_mutation:
                mutation_sites = {}
                for chain in chains:
                    mutation_sites[chain] = []
                for mut in mut_name.split('_')[-1].split(','):
                    chain = mut[1]
                    site = self._get_relative_site(pdb_name=mut_name, chain=chain, acutual_site=int(mut[2:-1]))
                    mutation_sites[chain].append(site)
                # {'A': [0, 1], 'B': [190, 191]}

                mutation_masks = {}
                for chain, id in chains.items():
                    mutation_mask = torch.zeros_like(chain_id[chain_id == id])
                    try:
                        mutation_mask[mutation_sites[chain]] = 1
                    except:
                        print(mut_name)
                        for chain in chains:
                            print(f"{chain}: {len(chain_id[chain_id == id])}")
                        print(mutation_sites)
                    mutation_masks[chain] = mutation_mask
                mutation_mask = torch.cat([mutation_masks[chain] for chain in mutation_masks]).to(torch.bool)

                if self.cut_interface and pair_dis_matrix is not None:
                    mut_dis_matrix = pair_dis_matrix[mutation_mask]
                else:
                    coord = wt_complex.coord
                    ca_wt = coord[:, 1]
                    ca_mut = ca_wt[mutation_mask]
                    mut_dis_matrix = pairwise_distances(ca_mut, ca_wt)
                mutation_surrounding_index = list(set(np.where(mut_dis_matrix < self.contact_threshold)[1]))
                mask[mutation_surrounding_index] = True

            wt_complex._mask_select(mask)
            mut_complex._mask_select(mask)

        wt_graph = self.build_graph(wt_complex)
        mut_graph = self.build_graph(mut_complex)

        # wt_graph = self.featurize_graph(wt_name)
        # mut_graph = self.featurize_graph(mut_name)

        y = torch.tensor(y, dtype=torch.float32)
        return mut_name, wt_graph, mut_graph, y
    
    def get_complex(self, name, jitter):
        with open(f"{self.feature_path}/seq/{name}.pkl", 'rb') as seq_file:
            seq = pkl.load(seq_file)
        chains = {chain: i for i, chain in enumerate(seq.keys())}
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
    
    def _get_relative_site(self, pdb_name: str, chain: str, acutual_site: int):
        pdb_filepath = f"{self.feature_path}/pdb/{pdb_name}.pdb"
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_name, pdb_filepath)
        model = structure[0]
        chain = model[chain]
        slice = [res.id[1] for res in chain.get_residues()]

        relative_site = slice.index(acutual_site)
        return relative_site

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
    def __init__(self, data: pd.DataFrame, feature_path="./data/", graph_mode="radius", radius=15, top_k=30,
                 cut_interface=True, cut_mutation=True, contact_threshold=10) -> None:
        super(SiameseProteinGraphDataset_prediction, self).__init__()

        self.data = data
        self.feature_path = feature_path
        self.graph_mode = graph_mode
        if self.graph_mode == "radius":
            self.radius = radius
        elif self.graph_mode == "knn":
            self.top_k = top_k
        self.cut_interface = cut_interface
        self.cut_mutation = cut_mutation
        self.contact_threshold = contact_threshold
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
        
        wt_complex = self.get_complex(wt_name)
        mut_complex = self.get_complex(mut_name)

        mut_name = mut_name if len(mut_name.split('_')) > 1 else wt_name
        chains = wt_complex.chains
        chain_id = wt_complex.chain_id
        
        # cut interface & mutation surrounding
        if self.cut_interface or self.cut_mutation:
            mask = np.zeros(len(wt_complex.seq)).astype('bool')

            # get interface_index
            if self.cut_interface:
                coord = wt_complex.coord
                ca_wt = coord[:, 1]
                pair_dis_matrix = pairwise_distances(ca_wt, ca_wt)

                contact_dis_matrix = pair_dis_matrix.copy()
                for chain, i in chains.items():
                    isin = np.isin(chain_id, i)
                    self_chain_mask = np.multiply.outer(isin, isin)
                    contact_dis_matrix[self_chain_mask] = 10000
                interface_index = list(set(np.where(contact_dis_matrix < self.contact_threshold)[0]))
                mask[interface_index] = True

            # get mutation_surrounding_index
            if self.cut_mutation:
                mutation_sites = {}
                for chain in chains:
                    mutation_sites[chain] = []
                for mut in mut_name.split('_')[-1].split(','):
                    chain = mut[1]
                    site = self._get_relative_site(pdb_name=mut_name, chain=chain, acutual_site=int(mut[2:-1]))
                    mutation_sites[chain].append(site)
                # {'A': [0, 1], 'B': [190, 191]}

                mutation_masks = {}
                for chain, id in chains.items():
                    mutation_mask = torch.zeros_like(chain_id[chain_id == id])
                    try:
                        mutation_mask[mutation_sites[chain]] = 1
                    except:
                        print(mut_name)
                        for chain in chains:
                            print(f"{chain}: {len(chain_id[chain_id == id])}")
                        print(mutation_sites)
                    mutation_masks[chain] = mutation_mask
                mutation_mask = torch.cat([mutation_masks[chain] for chain in mutation_masks]).to(torch.bool)

                if self.cut_interface and pair_dis_matrix is not None:
                    mut_dis_matrix = pair_dis_matrix[mutation_mask]
                else:
                    coord = wt_complex.coord
                    ca_wt = coord[:, 1]
                    ca_mut = ca_wt[mutation_mask]
                    mut_dis_matrix = pairwise_distances(ca_mut, ca_wt)
                mutation_surrounding_index = list(set(np.where(mut_dis_matrix < self.contact_threshold)[1]))
                mask[mutation_surrounding_index] = True

            wt_complex._mask_select(mask)
            mut_complex._mask_select(mask)

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

    def _get_relative_site(self, pdb_name: str, chain: str, acutual_site: int):
        pdb_filepath = f"{self.feature_path}/pdb/{pdb_name}.pdb"
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_name, pdb_filepath)
        model = structure[0]
        chain = model[chain]
        slice = [res.id[1] for res in chain.get_residues()]

        relative_site = slice.index(acutual_site)
        return relative_site


def get_data(dataset_path = "./data/dataset_processed.pt") -> pd.DataFrame:
    # dataset_train = torch.load(f"{data_path}/dataset_train.pt")
    # dataset_test = torch.load(f"{data_path}/dataset_test.pt")

    # dataset_train = SiameseProteinGraphDataset(dataset_train, feature_path="./data/", radius=15)
    # dataset_test = SiameseProteinGraphDataset(dataset_test, feature_path="./data/", radius=15)
    dataset = torch.load(dataset_path)

    return dataset











