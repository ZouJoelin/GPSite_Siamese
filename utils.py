import os
import subprocess
import random
import datetime

import numpy as np
import torch
from torchmetrics.functional import mean_squared_error, mean_absolute_error, spearman_corrcoef, pearson_corrcoef
from Bio import pairwise2


three_to_one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 
                'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 
                'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}

def get_clean_res_list(res_list, verbose=False, ensure_ca_exist=False, bfactor_cutoff=None):
    """ clean res_list """
    clean_res_list = []
    for res in res_list:
        hetero, resid, insertion = res.full_id[-1]
        if hetero == ' ':
            if res.resname not in three_to_one:
                if verbose:
                    print(res, "has non-standard resname")
                continue
            if (not ensure_ca_exist) or ('CA' in res):
                if bfactor_cutoff is not None:
                    ca_bfactor = float(res['CA'].bfactor)
                    if ca_bfactor < bfactor_cutoff:
                        continue
                clean_res_list.append(res)
        else:
            if verbose:
                print(res, res.full_id, "is hetero")
    return clean_res_list


def match_seq_wt2mut(wt_seq_file, mut_seq_file):
    print(f"verify sequence whether match: ")
    with open(wt_seq_file) as f:
        wt_seq = f.readline()
    with open(mut_seq_file) as f:
        mut_seq = f.readline()
    print(f"wt: {len(wt_seq)} - mut: {len(mut_seq)}")
    for i, (wt, mut) in enumerate(zip(wt_seq, mut_seq)):
        if wt != mut:
            print(f"{i+1}: {wt} -> {mut}")

def match_coord_wt2mut(wt_coord_file, mut_coord_file):
    print(f"verify coordinate whether match: ")
    wt_coord = torch.load(wt_coord_file)
    mut_coord = torch.load(mut_coord_file)
    print(f"wt: {wt_coord.shape} - mut: {mut_coord.shape}")

def match_ProtTrans_wt2mut(wt_ProtTrans_file, mut_ProtTrans_file):
    print(f"verify ProtTrans feature whether match: ")
    wt_ProtTrans = torch.load(wt_ProtTrans_file)
    mut_ProtTrans = torch.load(mut_ProtTrans_file)
    print(f"wt: {wt_ProtTrans.shape} - mut: {mut_ProtTrans.shape}")

def match_DSSP_wt2mut(wt_DSSP_file, mut_DSSP_file):
    print(f"verify DSSP feature whether match: ")
    wt_DSSP = torch.load(wt_DSSP_file)
    mut_DSSP = torch.load(mut_DSSP_file)
    print(f"wt: {wt_DSSP.shape} - mut: {mut_DSSP.shape}")

def match_wt2mut(wt_name, mut_name, feature_path):
    wt_seq_file = f"{feature_path}/seq/{wt_name}.txt"
    mut_seq_file = f"{feature_path}/seq/{mut_name}.txt"
    match_seq_wt2mut(wt_seq_file, mut_seq_file)

    wt_coord_file = f"{feature_path}/coord/{wt_name}.pt"
    mut_coord_file = f"{feature_path}/coord/{mut_name}.pt"
    match_coord_wt2mut(wt_coord_file, mut_coord_file)

    wt_ProtTrans_file = f"{feature_path}/ProtTrans/{wt_name}.pt"
    mut_ProtTrans_file = f"{feature_path}/ProtTrans/{mut_name}.pt"
    match_ProtTrans_wt2mut(wt_ProtTrans_file, mut_ProtTrans_file)

    wt_DSSP_file = f"{feature_path}/DSSP/{wt_name}.pt"
    mut_DSSP_file = f"{feature_path}/DSSP/{mut_name}.pt"
    match_DSSP_wt2mut(wt_DSSP_file, mut_DSSP_file)


def process_dssp(dssp_file):
    """ extract Second-Structure(SS) and relative-solvent-accessibility(RSA) from .dssp file.

    Args:
        dssp_file (string)

    Return:
        seq (string): AA sequence in .dssp file.
        dssp_feature (list<ndarray>): for each AA, use representation of (9,) array, 
        first element represent RSA, rest 8 elements represent SS_type in one-hot.
    """
    aa_type = "ACDEFGHIKLMNPQRSTVWY"
    SS_type = "HBEGITSC"
    rASA_std = [115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
                185, 160, 145, 180, 225, 115, 140, 155, 255, 230]

    with open(dssp_file, "r") as f:
        lines = f.readlines()

    seq = ""
    dssp_feature = []

    p = 0
    while lines[p].strip()[0] != "#":
        p += 1
    for i in range(p + 1, len(lines)):
        aa = lines[i][13]
        if aa == "!" or aa == "*":
            continue
        seq += aa
        SS = lines[i][16]
        if SS == " ":
            SS = "C"
        SS_vec = np.zeros(8)
        SS_vec[SS_type.find(SS)] = 1
        ASA = float(lines[i][34:38].strip())
        RSA = min(1, ASA / rASA_std[aa_type.find(aa)]) # relative solvent accessibility
        dssp_feature.append(np.concatenate((np.array([RSA]), SS_vec)))

    return seq, dssp_feature


def match_dssp(seq, dssp, ref_seq):
    """ pad dssp with np.zeros(9) if seq have gap according to ref_seq.
    
    Args:
        seq (string): dssp_seq.
        dssp (list<ndarray>)
        ref_seq: original sequence.

    Return:
        matched_dssp (list<ndarray>)
    """
    alignments = pairwise2.align.globalxx(ref_seq, seq)
    ref_seq = alignments[0].seqA
    seq = alignments[0].seqB

    padded_item = np.zeros(9)

    new_dssp = []
    for aa in seq:
        if aa == "-":
            new_dssp.append(padded_item)
        else:
            new_dssp.append(dssp.pop(0))

    matched_dssp = []
    for i in range(len(ref_seq)):
        if ref_seq[i] == "-":
            continue
        matched_dssp.append(new_dssp[i])

    return matched_dssp


def Seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def Write_log(logFile, text: str, isPrint=True):
    if isPrint:
        print(text)
    logFile.write(text)
    logFile.write('\n')


def get_GPUs_used_memory():
    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    output = subprocess.check_output(command.split()).decode("utf-8").strip().split("\n")[1:]
    gpu_memory = [int(line.split()[0]) for line in output]
    return gpu_memory


def Metric(pred: torch.tensor, target: torch.tensor):

    mse = mean_squared_error(pred, target)
    mae = mean_absolute_error(pred, target)
    abs_err_std = torch.std(torch.abs(pred - target))
    spearman_cc = spearman_corrcoef(pred, target)
    pearson_cc = pearson_corrcoef(pred, target)
    
    return mse, mae, abs_err_std, spearman_cc, pearson_cc
    

def metric2string(mse, mae, abs_err_std, spearman_cc, pearson_cc, pre_fix=""):
    if len(pre_fix) > 0:
        pre_fix += '_'
    metric_string = (f"{pre_fix}mse: {mse:.6f}, "
                     f"{pre_fix}mae: {mae:.6f}, "
                     f"{pre_fix}abs_err_std: {abs_err_std:.6f}, "
                     f"{pre_fix}scc: {spearman_cc:.6f}, " 
                     f"{pre_fix}pcc: {pearson_cc:.6f};" 
                     )

    return metric_string


def get_current_time():
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    return current_time


def get_current_timestamp():
    current_timestamp = datetime.datetime.now()

    return current_timestamp


def elapse_time(start: datetime.datetime, end: datetime.datetime):
    elapse = round((end - start).total_seconds())
    elapse = datetime.timedelta(seconds=elapse)

    return elapse
