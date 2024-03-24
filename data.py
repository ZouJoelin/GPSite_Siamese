import numpy as np
import pandas as pd

import torch
import torch.utils.data as data
import torch_geometric


class SiameseProteinGraphDataset(data.Dataset):
    def __init__(self, data: pd.DataFrame, feature_path="./data/", graph_mode="radius", radius=15, top_k=30) -> None:
        super(SiameseProteinGraphDataset, self).__init__()

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
        wt_name = entry['pdb'].split('_')[0]  # 1A4Y
        mut_name = entry['pdb'] +'_'+ entry['mutation']  # 1A4Y_A_B_DA435A
        y = entry['ddg']  # numpy.float64

        wt_graph = self.featurize_graph(wt_name)
        mut_graph = self.featurize_graph(mut_name)
        y = torch.tensor(y, dtype=torch.float32)

        return wt_graph, mut_graph, y

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
        
        graph_data = torch_geometric.data.Data(name=name, seq=seq, coord=coord, node_feat=pre_computed_node_feature, edge_index=edge_index)
        return graph_data


def get_data(data_path = "./data") -> [pd.DataFrame, pd.DataFrame]:
    dataset_train = torch.load(f"{data_path}/dataset_train.pt")
    dataset_test = torch.load(f"{data_path}/dataset_test.pt")

    # dataset_train = SiameseProteinGraphDataset(dataset_train, feature_path="./data/", radius=15)
    # dataset_test = SiameseProteinGraphDataset(dataset_test, feature_path="./data/", radius=15)

    return dataset_train, dataset_test











