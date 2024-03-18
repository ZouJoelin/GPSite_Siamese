import numpy as np
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


def match_wt2mut(wt_seq_file, mut_seq_file):
    with open(wt_seq_file) as f:
        wt_seq = f.readline()

    with open(mut_seq_file) as f:
        mut_seq = f.readline()
        
    for i, (wt, mut) in enumerate(zip(wt_seq, mut_seq)):
        if wt != mut:
            print(f"{i+1}: {wt} -> {mut}")
    

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



