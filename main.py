# import everything...
import os
import argparse
from datetime import datetime
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
parser = argparse.ArgumentParser(description='Train SiameseGPSite for PPI mutation affinity prediction.')

parser.add_argument("--data_path", type=str, default="./data/",
                    help="path where store dataset and feature.")
parser.add_argument("--output_path", type=str, default="./output/",
                    help="log your training process.")

parser.add_argument("--debug", action='store_true', default=False)
parser.add_argument("--train", action='store_true', default=False)
parser.add_argument("--test", action='store_true', default=False)

parser.add_argument("--gpu_id", type=int, default=0,
                    help="select GPU to train on.")
parser.add_argument("--num_workers", type=int, default=0,
                    help="to turn on multi-process data loading, set the argument num_workers to a positive integer.")

args = parser.parse_args()


# running configuration
data_path = args.data_path
output_path = args.output_path
gpu_id = args.gpu_id
num_workers = args.num_workers
use_parallel = False
graph_size_limit = 400000

# hyper-parameter
seed = 42
train_samples = 5000
batch_size_train = 8
batch_size_valid = batch_size_test = 4
lr = 1e-4
beta12 = (0.9, 0.999)
folds = 5 #3
epochs_num = 200 #5
graph_mode = "knn"
top_k = 30
patience = 10

if args.debug:
    graph_size_limit = 100000
    train_samples = 10
    batch_size = 2
    folds = 3
    epochs_num = 5

# fix random seed for training
Seed_everything(seed)


############### Finished preparing, start running ###############

# record output
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
output_root = f"{output_path}/{timestamp}"
if args.debug:
    output_root += "_debug"
os.makedirs(output_root, exist_ok=True)


# choose device
if torch.cuda.is_available():
    gpu_ids = list(range(torch.cuda.device_count()))
    print(f"Available GPUs: {gpu_ids}")
    device = torch.device("cuda", gpu_ids[gpu_id])
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


# get dataset
dataset, dataset_test = get_data(data_path)


# train & valid
# train parameters & select best parameters
if args.train:

    # backup source code
    output_src_path = f"{output_root}/src/"
    os.makedirs(output_src_path, exist_ok=True)
    os.system(f"cp *.py {output_src_path}/")

    # save best epoch's model parameters for each fold
    output_model_path = f"{output_root}/models/"
    os.makedirs(output_model_path, exist_ok=True)

    # reocrd loss_history in training
    output_loss_path = f"{output_root}/loss/"
    os.makedirs(output_loss_path, exist_ok=True)

    # reocrd metric_history in training
    output_metric_path = f"{output_root}/metrics/"
    os.makedirs(output_metric_path, exist_ok=True)

    log = open(f"{output_root}/train_valid.log", 'w', buffering=1)
    current_time = datetime.now().strftime("%m-%d %H:%M")
    Write_log(log, f"\n==================== Train & Validate with {folds}-Fold @{current_time} ====================")


    best_valid_metric = {"MSE": [], "MAE": [], "STD": [], "SCC": []}
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (index_train, index_valid) in enumerate(kf.split(dataset)):
        current_time = datetime.now().strftime("%m-%d %H:%M")
        Write_log(log, f"\n========== Fold {fold} @{current_time} ==========")

        # split out dataset_train
        dataset_train = dataset.iloc[index_train].reset_index(drop=True)
        dataset_train = SiameseProteinGraphDataset(dataset_train, feature_path=data_path, graph_mode=graph_mode, top_k=top_k)
        sampler = RandomSampler(dataset_train, replacement=True, num_samples=train_samples)
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size_train, sampler=sampler, shuffle=False, drop_last=True, num_workers=num_workers, prefetch_factor=2)
        Write_log(log, f"dataset_train: {len(dataset_train)} dataloader_train: {len(dataloader_train)}")  # total_data * (fold-1)/fold

        # split out dataset_valid
        if args.debug:
            index_valid = list(range(batch_size_valid*2))
        dataset_valid = dataset.iloc[index_valid].reset_index(drop=True)
        dataset_valid = SiameseProteinGraphDataset(dataset_valid, feature_path=data_path, graph_mode=graph_mode, top_k=top_k)
        dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size_valid, shuffle=False, drop_last=False, num_workers=num_workers, prefetch_factor=2)
        Write_log(log, f"dataset_valid: {len(dataset_valid)} dataloader_valid: {len(dataloader_valid)}")  # total_data * 1/fold

        # get empty model for each fold
        model = get_model().to(device)

        # choose optimizer, scheduler, loss_fn
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=beta12, weight_decay=1e-5, eps=1e-5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(dataloader_train), epochs=epochs_num)
        loss_fn = nn.MSELoss()

        # DataParallel
        """ToDO..."""
        if use_parallel and len(gpu_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
            optimizer = torch.nn.DataParallel(optimizer, device_ids=gpu_ids)
            scheduler = torch.nn.DataParallel(scheduler, device_ids=gpu_ids)
            Write_log(log, f"DataParallel with {len(gpu_ids)} GPUs")


        train_loss_history = []
        metric_history = {
            "train": {
                "MSE": [],  # Mean_Squared_Error
                "MAE": [],  # Mean_Absolute_Error
                "STD": [],  # Absolute_Error_STD
                "SCC": []   # Spearman_correlation_coefficient
            },
            "valid": {
                "MSE": [], "MAE": [], "STD": [], "SCC": [] 
            }
        }
        best_valid_mse = float('inf') 
        epochs_not_improving = 0
        for epoch in range(epochs_num):

            graph_size_seen_upmost = 0
            memory_used_upmost = 0

            # train
            model.train()
            pred_list = []
            y_list = []
            progress_bar = tqdm(enumerate(dataloader_train), total=len(dataloader_train), position=0, leave=True)
            for batch, (wt_graph, mut_graph, y) in progress_bar:
                optimizer.zero_grad()
                wt_graph, mut_graph, y = wt_graph.to(device), mut_graph.to(device), y.to(device)

                # skip too large protein, in case CUDA out of memory
                current_graph_size = wt_graph.edge_index.shape[1]
                if current_graph_size > graph_size_limit:
                    Write_log(log, (f"[Warning]: protein({wt_graph.name}) graph size({current_graph_size}) exceed pre-set limit {graph_size_limit} "
                                    f"(GPU{gpu_id}_memory_used_upmost: {memory_used_upmost} graph_size_seen_upmost: {graph_size_seen_upmost}"
                                    ))
                    continue
                # updata largest graph have seen
                if current_graph_size > graph_size_seen_upmost:
                    graph_size_seen_upmost = current_graph_size
                    memory_used_upmost = get_GPUs_used_memory()[gpu_id]

                pred = model(wt_graph, mut_graph)
                loss = loss_fn(pred, y)

                train_loss_history.append(loss.item())
                pred_list += pred.tolist()
                y_list += y.tolist()

                loss.backward()
                optimizer.step()
                scheduler.step()

                progress_bar.set_description(f"\033[31mloss: {loss:.4f}\033[0m")

            assert (len(pred_list) == len(y_list))
            train_mse, train_mae, train_std, train_scc = Metric(torch.tensor(np.array(pred_list)), torch.tensor(np.array(y_list)))
            metric_history["train"]["MSE"].append(train_mse.item())
            metric_history["train"]["MAE"].append(train_mae.item())
            metric_history["train"]["STD"].append(train_std.item())
            metric_history["train"]["SCC"].append(train_scc.item())


            # valid
            model.eval()
            pred_list = []
            y_list = []
            with torch.no_grad():
                for batch, (wt_graph, mut_graph, y) in enumerate(dataloader_valid):
                    wt_graph, mut_graph, y = wt_graph.to(device), mut_graph.to(device), y.to(device)
                    pred = model(wt_graph, mut_graph)

                    pred_list += pred.tolist()
                    y_list += y.tolist()

            assert (len(pred_list) == len(y_list))
            valid_mse, valid_mae, valid_std, valid_scc = Metric(torch.tensor(np.array(pred_list)), torch.tensor(np.array(y_list)))
            metric_history["valid"]["MSE"].append(valid_mse.item())
            metric_history["valid"]["MAE"].append(valid_mae.item())
            metric_history["valid"]["STD"].append(valid_std.item())
            metric_history["valid"]["SCC"].append(valid_scc.item())


            # record epoch progress
            if valid_mse.item() < best_valid_mse:
                # save current epoch model as best model of this fold.
                torch.save(model.state_dict(), f"{output_model_path}/model_fold{fold}.ckpt")
                best_valid_mse = valid_mse.item()
                best_valid_mae = valid_mae.item()
                best_valid_std = valid_std.item()
                best_valid_scc = valid_scc.item()
                epochs_not_improving = 0
                improve_or_not = ""
            else:
                epochs_not_improving += 1
                improve_or_not = f"No improvement +{epochs_not_improving}"
                

            # log this epoch
            Write_log(log, (f"Epoch[{epoch}] lr: {scheduler.get_last_lr()[0]}; "
                            f"{metric2string(train_mse, train_mae, train_std, train_scc, pre_fix='train')} "
                            f"{metric2string(valid_mse, valid_mae, valid_std, valid_scc, pre_fix='valid')} "
                            f"{improve_or_not}"
                            ))

            # early stop
            if epochs_not_improving > patience:
                break
        
        # best model of this fold
        best_valid_metric["MSE"].append(best_valid_mse)
        best_valid_metric["MAE"].append(best_valid_mae)
        best_valid_metric["STD"].append(best_valid_std)
        best_valid_metric["SCC"].append(best_valid_scc)
        Write_log(log, (f"\nFold[{fold}] Best model on dataset_valid: "
                        f"{metric2string(best_valid_mse, best_valid_mae, best_valid_std, best_valid_scc, pre_fix='best_valid')}"
                        ))

        # save train_loss_history of every batches of every epochs for each fold
        with open(f"{output_loss_path}/train_loss_fold{fold}.pkl", "wb") as loss_file:
            pickle.dump(train_loss_history, loss_file)
        
        # save metric_history of all epochs' training for each fold
        with open(f"{output_metric_path}/metrics_fold{fold}.pkl", "wb") as metrics_file:
            pickle.dump(metric_history, metrics_file)

        # finish this fold, clean gpu memory
        model.to("cpu")
        del dataloader_train, dataloader_valid
        torch.cuda.empty_cache()

    current_time = datetime.now().strftime("%m-%d %H:%M")
    Write_log(log, f"\n==================== Finish {fold}-Fold training & validating @{current_time} ====================")

    # average on cross_validation metric
    with open(f"{output_metric_path}/CV_metrics.pkl", "wb") as cv_metrics_file:
        pickle.dump(best_valid_metric, cv_metrics_file)

    cv_valid_mse = np.mean(best_valid_metric['MSE'])
    cv_valid_mae = np.mean(best_valid_metric['MAE'])
    cv_valid_std = np.mean(best_valid_metric['STD'])
    cv_valid_scc = np.mean(best_valid_metric['SCC'])
    Write_log(log, (f"Cross Validation mean metrics: "
                    f"{metric2string(cv_valid_mse, cv_valid_mae, cv_valid_std, cv_valid_scc, pre_fix='CV')}"
                    ))
    
    log.close()




# # test
# # final independent evaluation on each fold's model with dataset_test
# if args.test:


# models = [model[fold] for fold in folds]

# metric_test = 0
# y_list = []
# pred_list = []

# for batch in dataset_test:
#     x, y = x.to(device), y.to(device)

#     pred = [model(x) for model in models]
#     pred = pred.mean()
    
#     metric_test += Metric(pred, y)
#     pred_list.append(pred)
#     y_list.append(y)

# metric_test = metric_test.mean()

# # save metric and pred_y


# log("independent test: {metric_test}")



# log.close() 


