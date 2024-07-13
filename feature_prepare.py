import os
import glob
from tqdm import tqdm
import pickle as pkl
import numpy as np
import torch

import gc
from transformers import T5Tokenizer, T5EncoderModel

from Bio.PDB import PDBParser
from utils import *

data_pdb_path = "./data/skempi_v2/pdb/"
data_feature_root = "./data/skempi_split_chains/"
gpu = '1'


# extract sequence and coordinate
print("############### Extract sequence and coordinate ###############")

data_seq_path = data_feature_root + "/seq/"
os.makedirs(data_seq_path, exist_ok=True)
data_coord_path = data_feature_root + "/coord/"
os.makedirs(data_coord_path, exist_ok=True)

pdb_filepath_list = glob.glob(data_pdb_path + "*.pdb")
for pdb_filepath in tqdm(pdb_filepath_list):
    pdb_name = os.path.basename(pdb_filepath).split(".")[0]
    seqs = {}
    coords = {}
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_name, pdb_filepath)
    for chain in structure.get_chains():
        chain_id = chain.get_id()
        res_list = get_clean_res_list(chain.get_residues(), verbose=False, ensure_ca_exist=True)
        # ensure all res contains N, CA, C and O
        res_list = [res for res in res_list if (('N' in res) and ('CA' in res) and ('C' in res) and ('O' in res))]

        # extract seq
        seq = "".join([three_to_one.get(res.resname) for res in res_list])
        seqs[chain_id] = seq

        # extract coord
        coord = []
        for res in res_list:
            res_coord = []
            R_group = []
            for atom in res:
                if atom.name in ["N", "CA", "C", "O"]:
                    res_coord.append(atom.get_coord())
                else:
                    R_group.append(atom.get_coord())
            if len(R_group) == 0:
                R_group.append(res['CA'].get_coord())
            R_group = np.array(R_group).mean(axis=0)
            res_coord.append(R_group)
            coord.append(res_coord)
        coord = np.array(coord)  # convert list directly to tensor would be rather slow, suggest use ndarray as transition
        coord = torch.tensor(coord, dtype=torch.float32)
        coords[chain_id] = coord

    # print(pdb_name)
    # for chain_id in seqs:
    #     print(f"chain_id: {chain_id}")
    #     print(f"seq: {seqs[chain_id]}")
    #     print(f"coord: {coords[chain_id].shape}")

    # save to file
    seq_to_file = data_seq_path + pdb_name + ".pkl"
    with open(seq_to_file, "wb") as f:
        pkl.dump(seqs, f)
    coord_to_file = data_coord_path + pdb_name + ".pt"
    torch.save(coords, coord_to_file)
    
    
# extract ProtTrans feature
print("############### Extract ProtTrans feature ###############")

data_ProtTrans_raw_path = data_feature_root + "/ProtTrans_raw/"
os.makedirs(data_ProtTrans_raw_path, exist_ok=True)

ProtTrans_toolpath = "./tools/Prot-T5-XL-U50/"

# Load the vocabulary and ProtT5-XL-UniRef50 Model
tokenizer = T5Tokenizer.from_pretrained(ProtTrans_toolpath, do_lower_case=False)
model = T5EncoderModel.from_pretrained(ProtTrans_toolpath)
gc.collect()
# Load the model into the GPU if avilabile and switch to inference mode
device = torch.device('cuda:' + gpu if torch.cuda.is_available() and gpu else 'cpu')
model = model.to(device)
model = model.eval()

Max_protrans = []
Min_protrans = []

seq_filepath_list = glob.glob(data_seq_path + "*.pkl")
for seq_filepath in tqdm(seq_filepath_list):
    pdb_name = os.path.basename(seq_filepath).split(".")[0]
    ProtTrans_features_raw = {}
    with open(seq_filepath, 'rb') as seq_file:
        seqs = pkl.load(seq_file)
    for chain_id in seqs:
        seq = seqs[chain_id]

        batch_name_list = [pdb_name + "_" + chain_id]
        batch_seq_list = [" ".join(list(seq))]

        # Tokenize, encode sequences and load it into the GPU if possibile
        ids = tokenizer.batch_encode_plus(batch_seq_list, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        # Extracting sequences' features and load it into the CPU if needed
        with torch.no_grad():
            embedding = model(input_ids=input_ids,attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu()

        # Remove padding (\<pad>) and special tokens (\</s>) that is added by ProtT5-XL-UniRef50 model
        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emb = embedding[seq_num][:seq_len-1]
            # print(f"truncate padding: {embedding[seq_num].shape} -> {seq_emb.shape}")
            seq_emb = seq_emb.numpy()
            ProtTrans_features_raw[chain_id] = seq_emb

            Max_protrans.append(np.max(seq_emb, axis = 0))
            Min_protrans.append(np.min(seq_emb, axis = 0))

    # print(pdb_name)
    # for chain_id in seqs:
    #     print(f"chain_id: {chain_id}")
    #     print(f"ProtTrans_features_raw: {ProtTrans_features_raw[chain_id].shape}")

    # save to file
    ProtTrans_to_file = data_ProtTrans_raw_path + pdb_name + ".pt"
    torch.save(ProtTrans_features_raw, ProtTrans_to_file)


# normalize raw ProtTrans
print("############### Normalize ProtTrans feature ###############")
data_ProtTrans_path = data_feature_root + "/ProtTrans/"
os.makedirs(data_ProtTrans_path, exist_ok=True)

Min_protrans = np.min(np.array(Min_protrans), axis = 0)
Max_protrans = np.max(np.array(Max_protrans), axis = 0)
np.save(f"{data_feature_root}/Max_ProtTrans_repr.npy", Max_protrans)
np.save(f"{data_feature_root}/Min_ProtTrans_repr.npy", Min_protrans)

ProtTrans_filepath_list = glob.glob(data_ProtTrans_raw_path + "*.pt")
for ProtTrans_filepath in tqdm(ProtTrans_filepath_list):
    pdb_name = os.path.basename(ProtTrans_filepath).split(".")[0]
    ProtTrans_features = {}
    ProtTrans_features_raw = torch.load(ProtTrans_filepath)
    for chain_id in ProtTrans_features_raw:
        ProtTrans_feature_raw = ProtTrans_features_raw[chain_id]
        ProtTrans_feature = (ProtTrans_feature_raw - Min_protrans) / (Max_protrans - Min_protrans)
        ProtTrans_features[chain_id] = torch.tensor(ProtTrans_feature, dtype=torch.float32)

    # print(pdb_name)
    # for chain_id in ProtTrans_features:
    #     print(f"chain_id: {chain_id}")
    #     print(f"ProtTrans_features: {ProtTrans_features[chain_id].shape}")

    # save to file
    ProtTrans_to_file = data_ProtTrans_path + pdb_name + ".pt"
    torch.save(ProtTrans_features, ProtTrans_to_file)

os.system(f"rm -rf {data_ProtTrans_raw_path}")


# extract DSSP feature
print("############### Extract DSSP feature ###############")

data_DSSP_path = data_feature_root + "/DSSP/"
os.makedirs(data_DSSP_path, exist_ok=True)

dssp_toolpath = "./tools/mkdssp"

seq_filepath_list = glob.glob(data_seq_path + "*.pkl")
for seq_filepath in tqdm(seq_filepath_list):
    pdb_name = os.path.basename(seq_filepath).split(".")[0]
    dssp_features = {}
    seq = ''
    with open(seq_filepath, 'rb') as seq_file:
        seqs = pkl.load(seq_file)
    for chain_id in seqs:
        seq += seqs[chain_id]

    DSSP_to_file = data_DSSP_path + pdb_name + ".dssp"
    dssp_cmd = f"{dssp_toolpath} -i {data_pdb_path}/{pdb_name}.pdb -o {DSSP_to_file}"
    os.system(dssp_cmd)

    try:
        dssp_seq, dssp_matrix = process_dssp(DSSP_to_file)
        # dssp_seq: likely equal to original sequence
        # dssp_matrix (list<ndarray>): list of (1, 9) vector, length of dssp_seq
        if dssp_seq != seq:
            dssp_matrix = match_dssp(dssp_seq, dssp_matrix, seq)
        
        start = 0
        for chain_id in seqs:
            dssp_features[chain_id] = torch.tensor(np.array(dssp_matrix[start: start+len(seqs[chain_id])]), dtype=torch.float32)
            start += len(seqs[chain_id])
        
        # print(pdb_name)
        # for chain_id in dssp_features:
        #     print(f"chain_id: {chain_id}")
        #     print(f"dssp_features: {dssp_features[chain_id].shape}")

        DSSP_to_file = data_DSSP_path + pdb_name + ".pt"
        torch.save(dssp_features, DSSP_to_file)
        # shape(AA_len, 9)
        os.system(f"rm -rf {data_DSSP_path}/*.dssp")
    except:
        print(f"Wrong entry prompt: $ {dssp_cmd}")
        continue
















