
import numpy as np


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


def match_wt2mut(wt_seq_file, mut_seq_file):
    with open(wt_seq_file) as f:
        wt_seq = f.readline()

    with open(mut_seq_file) as f:
        mut_seq = f.readline()
        
    for i, (wt, mut) in enumerate(zip(wt_seq, mut_seq)):
        if wt != mut:
            print(f"{i+1}: {wt} -> {mut}")
    






