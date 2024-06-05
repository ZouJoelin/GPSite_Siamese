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
parser = argparse.ArgumentParser(description='Test SiameseGPSite with labeled dataset.')

parser.add_argument("--dataset_path", type=str, default="./data/mavedb/dataset_processed.pt",
                    help="file path where store dataset.")
parser.add_argument("--feature_path", type=str, default="./data/mavedb/",
                    help="root path where store feature.")
parser.add_argument("--dataset_name", type=str, required=True,
                    help="name of dataset.")
parser.add_argument("--model_path", type=str, default="./models/",
                    help="path to load trained_model.")
parser.add_argument("--output_path", type=str, default="./test/",
                    help="log your training process.")

parser.add_argument("--mean_on_models", action='store_true', default=False)

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
mean_on_models = args.mean_on_models
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

# record pred_target pairs while running
output_pred_target_path = f"{output_path}/pred_target/"
os.makedirs(output_pred_target_path, exist_ok=True)

# log initial information
log = open(f"{output_path}/test.log", 'w', buffering=1)
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

# get models
model_list = []
models_filepath_list = glob.glob(f"{model_path}/model_fold*.ckpt")
models_filepath_list.sort()
for model_filepath in models_filepath_list:
    model = get_model().to(device)
    checkpoint = torch.load(model_filepath, device)
    model.load_state_dict(checkpoint)
    model.eval()
    model_list.append(model)
folds_num = len(model_list)
Write_log(log, f"finished models count: {folds_num}")

# get dataset
dataset = get_data(dataset_path)


# predict
test_pred_y = {"name": [], "pred": [], "target": []}
test_metrics = {"MSE": [], "MAE": [], "STD": [], "SCC": [], "PCC": []}

if not mean_on_models:
    Write_log(log, "Using each model to predict each fold.")

    for fold, model in enumerate(model_list):
        Write_log(log, f"\n========== Fold {fold} @{get_current_time()} ========== ")

        # select out dataset_test for this fold
        splits_num = dataset["split"].max() + 1
        splits_per_fold = int(splits_num / folds_num)
        
        index_test = fold
        index_test = set(range(index_test*splits_per_fold, (index_test+1)*splits_per_fold))
        dataset_test = dataset[dataset["split"].isin(index_test)].reset_index(drop=True)
        dataset_test = SiameseProteinGraphDataset(dataset_test, feature_path=feature_path, graph_mode=graph_mode, top_k=top_k)
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size_test, shuffle=False, drop_last=False, num_workers=num_workers, prefetch_factor=2, pin_memory=pin_memory)
        Write_log(log, f"dataset_test: {len(dataset_test)} dataloader_test: {len(dataloader_test)}")

        name_list = []
        pred_list = []
        y_list = []
        with torch.no_grad():
            for batch, (name, wt_graph, mut_graph, y) in tqdm(enumerate(dataloader_test), total=len(dataloader_test)):
                wt_graph, mut_graph, y = wt_graph.to(device), mut_graph.to(device), y.to(device)
                pred = model(wt_graph, mut_graph)

                name_list += name
                pred_list.append(pred)
                y_list.append(y)

            pred_list = torch.hstack(pred_list).tolist()
            y_list = torch.hstack(y_list).tolist()
            # collect total test_pred_y pairs
            with open(f"{output_pred_target_path}/test_pred_target_fold_{fold}.pkl", "wb") as pred_y_file:
                pred_target = {"name": name_list, "pred": pred_list, "target": y_list}
                pickle.dump(pred_target, pred_y_file)

            test_pred_y["name"].append(name_list)
            test_pred_y["pred"].append(pred_list)
            test_pred_y["target"].append(y_list)

            assert (len(test_pred_y["pred"]) == len(test_pred_y["target"]))
            test_mse, test_mae, test_std, test_scc, test_pcc = Metric(pred_list, y_list)
    
        test_metrics["MSE"].append(test_mse.item())
        test_metrics["MAE"].append(test_mae.item())
        test_metrics["STD"].append(test_std.item())
        test_metrics["SCC"].append(test_scc.item())
        test_metrics["PCC"].append(test_pcc.item())
        Write_log(log, (f"Fold[{fold}] Best model on dataset_test: "
                        f"{metric2string(test_mse, test_mae, test_std, test_scc, test_pcc, pre_fix='test')}"
                        ))
        
    Write_log(log, f"\n\n==================== Finish {folds_num}-Fold  @{get_current_time()} ==================== ")

    # evaluate on all test_pred_y pairs
    all_test_mse, all_test_mae, all_test_std, all_test_scc, all_test_pcc = Metric(torch.hstack([torch.tensor(l) for l in test_pred_y["pred"]]).tolist(), 
                                                                                torch.hstack([torch.tensor(l) for l in test_pred_y["target"]]).tolist())
    Write_log(log, (f"Independent Test metrics on all test_pred_y: "
                    f"{metric2string(all_test_mse, all_test_mae, all_test_std, all_test_scc, all_test_pcc, pre_fix='all_test')}"
                    ))
    
else:
    Write_log(log, "Using mean on models to predict.")

    dataset_test = SiameseProteinGraphDataset(dataset, feature_path=feature_path, graph_mode=graph_mode, top_k=top_k)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size_test, shuffle=False, drop_last=False, num_workers=num_workers, prefetch_factor=2, pin_memory=pin_memory)
    Write_log(log, f"dataset_test: {len(dataset_test)} dataloader: {len(dataloader_test)}")

    with torch.no_grad():
        for batch, (name, wt_graph, mut_graph, y) in tqdm(enumerate(dataloader_test), total=len(dataloader_test)):
            wt_graph, mut_graph, y = wt_graph.to(device), mut_graph.to(device), y.to(device)
            pred = [model(wt_graph, mut_graph) for model in model_list]
            pred = torch.vstack(pred).mean(dim=0)

            test_pred_y["name"] += name
            test_pred_y["pred"].append(pred)
            test_pred_y["target"].append(y)
        
        test_pred_y["pred"] = torch.hstack(test_pred_y["pred"]).tolist()
        test_pred_y["target"] = torch.hstack(test_pred_y["target"]).tolist()
        assert (len(test_pred_y["pred"]) == len(test_pred_y["target"]))
        test_mse, test_mae, test_std, test_scc, test_pcc = Metric(test_pred_y["pred"], test_pred_y["target"])
    test_metrics["MSE"].append(test_mse.item())
    test_metrics["MAE"].append(test_mae.item())
    test_metrics["STD"].append(test_std.item())
    test_metrics["SCC"].append(test_scc.item())
    test_metrics["PCC"].append(test_pcc.item())
    Write_log(log, metric2string(test_mse, test_mae, test_std, test_scc, test_pcc, pre_fix='test'))


# save pred-y pairs
with open(f"{output_path}/test_pred_y.pkl", "wb") as pred_y_file:
    pickle.dump(test_pred_y, pred_y_file)

# save test metrics
with open(f"{output_path}/test_metrics.pkl", "wb") as test_metrics_file:
    pickle.dump(test_metrics, test_metrics_file)


Write_log(log, f"\n==================== Finish testing @{get_current_time()} ====================")

log.close() 

