# import everything...
import os
import sys
import glob
import argparse
from tqdm import tqdm

import pickle
import numpy as np
import torch
from torch.utils.data import RandomSampler
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold

from data import *
from model import *
from utils import *


# parse argument
parser = argparse.ArgumentParser(description='Test SiameseGPSite with external dataset.')

parser.add_argument("--dataset_path", type=str, default="./data/mavedb/dataset_processed.pt",
                    help="file path where store dataset.")
parser.add_argument("--feature_path", type=str, default="./data/mavedb/",
                    help="root path where store feature.")
parser.add_argument("--dataset_name", type=str, required=True,
                    help="name of dataset.")
parser.add_argument("--model_path", type=str, default="./models/",
                    help="path to load trained_model.")
parser.add_argument("--output_path", type=str, default="./predict/",
                    help="log your training process.")


parser.add_argument("--gpu_id", type=int, default=0,
                    help="select GPU to train on.")
parser.add_argument("--num_workers", type=int, default=8,
                    help="to turn on CPU's multi-process to load data, set the argument num_workers to a positive integer.")
parser.add_argument("--pin_memory", action='store_true', default=True,
                    help="if True, the data loader will copy Tensors direatly into device/CUDA pinned memory before returning them")

args = parser.parse_args()


# running configuration
dataset_path = args.dataset_path
feature_path = args.feature_path
dataset_name = args.dataset_name
model_path = args.model_path
output_path = args.output_path
gpu_id = args.gpu_id
num_workers = args.num_workers
pin_memory = args.pin_memory
use_parallel = False
seed = 42

# hyper-parameter
hyper_para = {
    'batch_size_test': 4,
    'graph_mode': "knn",
    'top_k': 30,
}

batch_size_test = hyper_para["batch_size_test"]
graph_mode = hyper_para["graph_mode"]
top_k = hyper_para["top_k"]

# fix random seed for training
Seed_everything(seed)


############### Finished preparing, start running ###############

output_path = f"{output_path}/{dataset_name}/"
os.makedirs(output_path, exist_ok=True)

# reocrd d_embedding in test
output_d_embedding_path = f"{output_path}/d_embedding/"
os.makedirs(output_d_embedding_path, exist_ok=True)

# log initial information
log = open(f"{output_path}/predict.log", 'w', buffering=1)
Write_log(log, ' '.join(sys.argv))
Write_log(log, f"{hyper_para}\n")
Write_log(log, f"\n==================== External Test @{get_current_time()} ====================")

# choose device
if torch.cuda.is_available():
    gpu_ids = list(range(torch.cuda.device_count()))
    Write_log(log, f"Available GPUs: {gpu_ids}")
    device = torch.device("cuda", gpu_ids[gpu_id])
else:
    device = torch.device("cpu")
Write_log(log, f"Using device: {device}")

# get dataset
dataset_test = get_data(dataset_path)
dataset_test = SiameseProteinGraphDataset(dataset_test, feature_path=feature_path, graph_mode=graph_mode, top_k=top_k)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size_test, shuffle=False, drop_last=False, num_workers=num_workers, prefetch_factor=2)
Write_log(log, f"dataset_test: {len(dataset_test)} dataloader: {len(dataloader_test)}")

# get models
model_list = []
models_filepath_list = glob.glob(f"{model_path}/model_fold*.ckpt")
for model_filepath in models_filepath_list:
    model = get_model().to(device)
    checkpoint = torch.load(model_filepath, device)
    model.load_state_dict(checkpoint)
    model.eval()
    model_list.append(model)
Write_log(log, f"finished models count: {len(model_list)}")

# predict
test_pred_y = {"pred": [], "y": []}
test_metrics = {}
with torch.no_grad():
    for batch, (wt_graph, mut_graph, y) in tqdm(enumerate(dataloader_test), total=len(dataloader_test)):
        wt_graph, mut_graph, y = wt_graph.to(device), mut_graph.to(device), y.to(device)
        pred = [model(wt_graph, mut_graph) for model in model_list]
        pred = torch.vstack(pred).mean(dim=0)
        test_pred_y["pred"].append(pred)
        test_pred_y["y"].append(y)
    
    test_pred_y["pred"] = torch.hstack(test_pred_y["pred"]).tolist()
    test_pred_y["y"] = torch.hstack(test_pred_y["y"]).tolist()
    assert (len(test_pred_y["pred"]) == len(test_pred_y["y"]))
    test_mse, test_mae, test_std, test_scc, test_pcc = Metric(test_pred_y["pred"], test_pred_y["y"])
test_metrics["MSE"] = test_mse.item()
test_metrics["MAE"] = test_mae.item()
test_metrics["STD"] = test_std.item()
test_metrics["SCC"] = test_scc.item()
test_metrics["PCC"] = test_pcc.item()

# save pred-y pairs
with open(f"{output_path}/test_pred_y.pkl", "wb") as pred_y_file:
    pickle.dump(test_pred_y, pred_y_file)

# save test metrics
with open(f"{output_path}/test_metrics.pkl", "wb") as test_metrics_file:
    pickle.dump(test_metrics, test_metrics_file)


Write_log(log, metric2string(test_mse, test_mae, test_std, test_scc, test_pcc, pre_fix='test'))

Write_log(log, f"\n==================== Finish testing @{get_current_time()} ====================")

log.close() 

