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

import wandb
wandb.login()


# parse argument
parser = argparse.ArgumentParser(description='Train SiameseGPSite for PPI mutation affinity prediction.')

parser.add_argument("--data_path", type=str, default="./data/",
                    help="path where store dataset and feature.")
parser.add_argument("--output_path", type=str, default="./output/",
                    help="log your training process.")

pretrain_model_mask = parser.add_mutually_exclusive_group()
pretrain_model_mask.add_argument("--checkpoint_model_path", type=str, default=None,
                    help="continue from checkpoint.")
pretrain_model_mask.add_argument("--pretrained_model_path", type=str, default=None,
                    help="transfer learning from GPSite.")

parser.add_argument("--data_augment", action='store_true', default=False,
                    help="train data augment by flip wt and mut.")

parser.add_argument("--run_id", type=str, required=True, default=None)
parser.add_argument("--debug", action='store_true', default=False)
parser.add_argument("--re_run", type=str, default=None)

parser.add_argument("--gpu_id", type=int, default=0,
                    help="select GPU to train on.")
parser.add_argument("--num_workers", type=int, default=8,
                    help="to turn on CPU's multi-process to load data, set the argument num_workers to a positive integer.")
parser.add_argument("--pin_memory", action='store_true', default=True,
                    help="if True, the data loader will copy Tensors direatly into device/CUDA pinned memory before returning them")

args = parser.parse_args()


# running configuration
data_path = args.data_path
output_path = args.output_path
checkpoint_model_path = args.checkpoint_model_path
pretrained_model_path = args.pretrained_model_path
data_augment = args.data_augment
run_id = args.run_id
gpu_id = args.gpu_id
re_run = args.re_run
re_run = [int(i) for i in re_run.split(',')] if re_run else None
num_workers = args.num_workers
pin_memory = args.pin_memory
use_parallel = False
seed = 42

# hyper-parameter
hyper_para = {
    'train_samples': 5000,  # 5761*0.8
    'batch_size_train': 8,
    'batch_size_test': 4,
    'lr': 1e-4,
    'beta12': (0.9, 0.999),
    'folds_num': 5,
    'epochs_num': 250,
    'graph_size_limit': 400000,
    'graph_mode': "knn",
    'top_k': 30,
    'patience': 75,
}

hyper_para_debug = {
    'train_samples': 8,
    'batch_size_train': 2,
    'batch_size_test': 2,
    'lr': 1e-4,
    'beta12': (0.9, 0.999),
    'folds_num': 3,
    'epochs_num': 3,
    'graph_size_limit': 100000,
    'graph_mode': "knn",
    'top_k': 30,
    'patience': 5,
}
if args.debug:
    hyper_para = hyper_para_debug

train_samples = hyper_para["train_samples"]*2 if data_augment else hyper_para["train_samples"]
batch_size_train = hyper_para["batch_size_train"]
batch_size_valid = batch_size_test = hyper_para["batch_size_test"]
lr = hyper_para["lr"]
beta12 = hyper_para["beta12"]
folds_num = hyper_para["folds_num"]
epochs_num = hyper_para["epochs_num"]
graph_size_limit = hyper_para["graph_size_limit"]
graph_mode = hyper_para["graph_mode"]
top_k = hyper_para["top_k"]
patience = hyper_para["patience"]


# fix random seed for training
Seed_everything(seed)


############### Finished preparing, start running ###############

# record output
if args.debug:
    run_id += "_debug"
output_root = f"{output_path}/{run_id}"
os.makedirs(output_root, exist_ok=True)

# backup source code
output_src_path = f"{output_root}/src/"
os.makedirs(output_src_path, exist_ok=True)
os.system(f"cp *.py {output_src_path}/")

# save best epoch's model parameters for each fold
output_models_path = f"{output_root}/models/"
os.makedirs(output_models_path, exist_ok=True)

# reocrd loss_history in training
output_loss_path = f"{output_root}/loss/"
os.makedirs(output_loss_path, exist_ok=True)

# reocrd metric_history in train, valid, test
output_metric_path = f"{output_root}/metrics/"
os.makedirs(output_metric_path, exist_ok=True)

# reocrd d_embedding in test
output_d_embedding_path = f"{output_root}/d_embedding/"
os.makedirs(output_d_embedding_path, exist_ok=True)


# log initial information
log = open(f"{output_root}/{run_id}.log", 'w', buffering=1)
Write_log(log, ' '.join(sys.argv))
Write_log(log, f"{hyper_para}\n")


# choose device
if torch.cuda.is_available():
    gpu_ids = list(range(torch.cuda.device_count()))
    Write_log(log, f"Available GPUs: {gpu_ids}")
    device = torch.device("cuda", gpu_ids[gpu_id])
else:
    device = torch.device("cpu")
Write_log(log, f"Using device: {device}")


# get dataset
dataset = get_data(data_path)


# K-Fold
Write_log(log, f"\n\n==================== Start {folds_num}-Fold  @{get_current_time()} ==================== ")

best_valid_metrics = {"MSE": [], "MAE": [], "STD": [], "SCC": [], "PCC": []}
all_test_metrics = {"MSE": [], "MAE": [], "STD": [], "SCC": [], "PCC": []}
valid_pred_y = {"pred": [], "y": []}
test_pred_y = {"pred": [], "y": []}

index = list(range(len(dataset)))
index_k_split = np.array_split(index, folds_num)
for fold in range(folds_num):
    Write_log(log, f"\n========== Fold {fold} @{get_current_time()} ========== ")
    if re_run and fold not in re_run:
        continue

    name = f"{run_id}_fold_{fold}" if not re_run else f"{run_id}_fold_{fold}*"
    wandb.init(project="GPSite_siamese", group=f"{run_id}", name=name)
    wandb.config.update(hyper_para)

    # split dataset
    index_train = set(range(folds_num))
    index_test = fold
    index_valid = (fold+1)%folds_num
    index_train.remove(index_test)
    index_train.remove(index_valid)

    index_test = index_k_split[index_test]
    index_valid = index_k_split[index_valid]
    index_train = np.concatenate([index_k_split[j] for j in index_train])
    if args.debug:
        index_valid = list(range(batch_size_valid*2+1))
        index_test = list(range(batch_size_test*2))

    # select out dataset_train
    dataset_train = dataset.iloc[index_train].reset_index(drop=True)
    # data augment
    if data_augment:
        dataset_train_flip = dataset_train.copy()
        dataset_train_flip["wt_name"], dataset_train_flip["mut_name"] = dataset_train_flip["mut_name"], dataset_train_flip["wt_name"]
        dataset_train_flip["ddg"] = -dataset_train_flip["ddg"]
        dataset_train = pd.concat([dataset_train, dataset_train_flip], ignore_index=True)

    dataset_train = SiameseProteinGraphDataset(dataset_train, feature_path=data_path, graph_mode=graph_mode, top_k=top_k)
    sampler = RandomSampler(dataset_train, replacement=True, num_samples=train_samples)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size_train, sampler=sampler, shuffle=False, drop_last=True, num_workers=num_workers, prefetch_factor=2, pin_memory=pin_memory)
    Write_log(log, f"dataset_train: {len(dataset_train)} dataloader_train: {len(dataloader_train)}")

    # select out dataset_valid        
    dataset_valid = dataset.iloc[index_valid].reset_index(drop=True)
    dataset_valid = SiameseProteinGraphDataset(dataset_valid, feature_path=data_path, graph_mode=graph_mode, top_k=top_k)
    dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size_valid, shuffle=False, drop_last=False, num_workers=num_workers, prefetch_factor=2, pin_memory=pin_memory)
    Write_log(log, f"dataset_valid: {len(dataset_valid)} dataloader_valid: {len(dataloader_valid)}")

    # select out dataset_test
    dataset_test = dataset.iloc[index_test].reset_index(drop=True)
    dataset_test = SiameseProteinGraphDataset(dataset_test, feature_path=data_path, graph_mode=graph_mode, top_k=top_k)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size_test, shuffle=False, drop_last=False, num_workers=num_workers, prefetch_factor=2, pin_memory=pin_memory)
    Write_log(log, f"dataset_test: {len(dataset_test)} dataloader_test: {len(dataloader_test)}")


    # get empty model for each fold
    model = get_model().to(device)

    # load pretrained parameters
    if checkpoint_model_path:
        checkpoint = torch.load(checkpoint_model_path, device)
        model.load_state_dict(checkpoint)
    elif pretrained_model_path:
        pretrained_dict = torch.load(pretrained_model_path, device)
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {layer: weight for layer, weight in pretrained_dict.items() if layer in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)


    # choose optimizer, scheduler, loss_fn
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=beta12)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=beta12, weight_decay=1e-5, eps=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(dataloader_train), epochs=epochs_num)
    loss_fn = nn.MSELoss()


    # DataParallel
    """ToDO..."""
    if use_parallel and len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        optimizer = torch.nn.DataParallel(optimizer, device_ids=gpu_ids)
        scheduler = torch.nn.DataParallel(scheduler, device_ids=gpu_ids)
        Write_log(log, f"DataParallel with {len(gpu_ids)} GPUs")


    # train & valid
    # train parameters & select best parameters
    train_loss_history = []
    metric_history = {
        "train": {"MSE": [], "MAE": [], "STD": [], "SCC": [], "PCC": []},
        "valid": {"MSE": [], "MAE": [], "STD": [], "SCC": [], "PCC": []}
    }  # Mean_Squared_Error, Mean_Absolute_Error, Absolute_Error_STD, Spearman_Correlation_Coefficient, Pearson_Correlation_Coefficient
    best_valid_pred_y = {}
    best_valid_pcc = float('-inf') 
    epochs_not_improving = 0
    for epoch in range(epochs_num):
        start = get_current_timestamp()

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
                Write_log(log, (f"[Warning] protein({wt_graph.name}) graph size({current_graph_size}) exceed pre-set limit {graph_size_limit} "
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
            pred_list.append(pred)
            y_list.append(y)

            loss.backward()
            optimizer.step()
            scheduler.step()

            progress_bar.set_description(f"\033[31mloss: {loss:.4f}\033[0m")
        
        pred_list = torch.hstack(pred_list).tolist()
        y_list = torch.hstack(y_list).tolist()
        assert (len(pred_list) == len(y_list))
        train_mse, train_mae, train_std, train_scc, train_pcc = Metric(pred_list, y_list)
        metric_history["train"]["MSE"].append(train_mse.item())
        metric_history["train"]["MAE"].append(train_mae.item())
        metric_history["train"]["STD"].append(train_std.item())
        metric_history["train"]["SCC"].append(train_scc.item())
        metric_history["train"]["PCC"].append(train_pcc.item())


        # valid
        model.eval()
        pred_list = []
        y_list = []
        with torch.no_grad():
            for batch, (wt_graph, mut_graph, y) in enumerate(dataloader_valid):
                wt_graph, mut_graph, y = wt_graph.to(device), mut_graph.to(device), y.to(device)
                pred = model(wt_graph, mut_graph)

                pred_list.append(pred)
                y_list.append(y)

            pred_list = torch.hstack(pred_list).tolist()
            y_list = torch.hstack(y_list).tolist()
            assert (len(pred_list) == len(y_list))
            valid_mse, valid_mae, valid_std, valid_scc, valid_pcc = Metric(pred_list, y_list)
        metric_history["valid"]["MSE"].append(valid_mse.item())
        metric_history["valid"]["MAE"].append(valid_mae.item())
        metric_history["valid"]["STD"].append(valid_std.item())
        metric_history["valid"]["SCC"].append(valid_scc.item())
        metric_history["valid"]["PCC"].append(valid_pcc.item())


        # record epoch progress
        if valid_pcc.item() > best_valid_pcc:
            # save current epoch model as best model of this fold.
            torch.save(model.cpu().state_dict(), f"{output_models_path}/modeling_fold{fold}.ckpt")
            model.to(device)
            best_valid_mse = valid_mse.item()
            best_valid_mae = valid_mae.item()
            best_valid_std = valid_std.item()
            best_valid_scc = valid_scc.item()
            best_valid_pcc = valid_pcc.item()
            best_valid_pred_y["pred"] = pred_list
            best_valid_pred_y["y"] = y_list
            epochs_not_improving = 0
            improve_or_not = ""
        else:
            epochs_not_improving += 1
            improve_or_not = f"No improvement +{epochs_not_improving}"


        end = get_current_timestamp()
        spent_time = elapse_time(start, end)
        # log this epoch
        Write_log(log, (f"Epoch[{epoch}] spent_time: {spent_time} lr: {scheduler.get_last_lr()[0]:.6e}; "
                        f"{metric2string(train_mse, train_mae, train_std, train_scc, train_pcc, pre_fix='train')} "
                        f"{metric2string(valid_mse, valid_mae, valid_std, valid_scc, valid_pcc, pre_fix='valid')} "
                        f"{improve_or_not}"
                        ))
        wandb.log({
            "train": {"MSE": train_mse, "MAE": train_mae, "STD": train_std, "SCC": train_scc, "PCC": train_pcc},
            "valid": {"MSE": valid_mse, "MAE": valid_mae, "STD": valid_std, "SCC": valid_scc, "PCC": valid_pcc},
            "spent_time": int(spent_time.seconds)
        })

        # early stop
        if epochs_not_improving >= patience:
            break

    wandb.finish()

    # save train_loss_history of every batches of every epochs for each fold
    with open(f"{output_loss_path}/train_loss_fold{fold}.pkl", "wb") as loss_file:
        pickle.dump(train_loss_history, loss_file)

    # save metric_history of all epochs' training for each fold
    with open(f"{output_metric_path}/metrics_fold{fold}.pkl", "wb") as metrics_file:
        pickle.dump(metric_history, metrics_file)
        

    # best model on dataset_valid of this fold
    os.system(f"mv {output_models_path}/modeling_fold{fold}.ckpt {output_models_path}/model_fold{fold}.ckpt")
    best_valid_metrics["MSE"].append(best_valid_mse)
    best_valid_metrics["MAE"].append(best_valid_mae)
    best_valid_metrics["STD"].append(best_valid_std)
    best_valid_metrics["SCC"].append(best_valid_scc)
    best_valid_metrics["PCC"].append(best_valid_pcc)
    # collect total valid_pred_y pairs
    valid_pred_y["pred"].append(torch.tensor(best_valid_pred_y["pred"]))
    valid_pred_y["y"].append(torch.tensor(best_valid_pred_y["y"]))

    Write_log(log, (f"\nFold[{fold}] Best model on dataset_valid: "
                    f"{metric2string(best_valid_mse, best_valid_mae, best_valid_std, best_valid_scc, best_valid_pcc, pre_fix='best_valid')}"
                    ))


    # test
    # independent evaluation on dataset_test of this fold

    # get best model of this fold
    model_filepath =f"{output_models_path}/model_fold{fold}.ckpt"
    model = get_model(group="test", d_embedding_path=output_d_embedding_path).to(device)
    model.load_state_dict(torch.load(model_filepath, device))
    model.eval()

    pred_list = []
    y_list = []
    with torch.no_grad():
        for batch, (wt_graph, mut_graph, y) in tqdm(enumerate(dataloader_test), total=len(dataloader_test)):
            wt_graph, mut_graph, y = wt_graph.to(device), mut_graph.to(device), y.to(device)
            pred = model(wt_graph, mut_graph)

            pred_list.append(pred)
            y_list.append(y)

        pred_list = torch.hstack(pred_list).tolist()
        y_list = torch.hstack(y_list).tolist()
        # collect total test_pred_y pairs
        test_pred_y["pred"].append(torch.tensor(pred_list))
        test_pred_y["y"].append(torch.tensor(y_list))

        assert (len(pred_list) == len(y_list))
        test_mse, test_mae, test_std, test_scc, test_pcc = Metric(pred_list, y_list)
    all_test_metrics["MSE"].append(test_mse.item())
    all_test_metrics["MAE"].append(test_mae.item())
    all_test_metrics["STD"].append(test_std.item())
    all_test_metrics["SCC"].append(test_scc.item())
    all_test_metrics["PCC"].append(test_pcc.item())
    Write_log(log, (f"Fold[{fold}] Best model on dataset_test: "
                    f"{metric2string(test_mse, test_mae, test_std, test_scc, test_pcc, pre_fix='test')}"
                    ))


Write_log(log, f"\n\n==================== Finish {folds_num}-Fold  @{get_current_time()} ==================== ")

# evaluate on all test_pred_y pairs
all_test_mse, all_test_mae, all_test_std, all_test_scc, all_test_pcc = Metric(torch.hstack(test_pred_y["pred"]).tolist(), 
                                                                              torch.hstack(test_pred_y["y"]).tolist())
Write_log(log, (f"Independent Test metrics on all test_pred_y: "
                f"{metric2string(all_test_mse, all_test_mae, all_test_std, all_test_scc, all_test_pcc, pre_fix='all_test')}"
                ))


# save k-fold valid_metrics & test_metrics
with open(f"{output_metric_path}/CV_metrics.pkl", "wb") as cv_metrics_file:
    pickle.dump(best_valid_metrics, cv_metrics_file)

with open(f"{output_metric_path}/all_test_metrics.pkl", "wb") as test_metrics_file:
    pickle.dump(all_test_metrics, test_metrics_file)

# save valid_pred_y pairs & test_pred_y pairs
with open(f"{output_root}/valid_pred_y.pkl", "wb") as pred_y_file:
    pickle.dump(valid_pred_y, pred_y_file)

with open(f"{output_root}/test_pred_y.pkl", "wb") as pred_y_file:
    pickle.dump(test_pred_y, pred_y_file)


log.close() 


