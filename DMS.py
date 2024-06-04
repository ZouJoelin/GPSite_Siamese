
import os
from tqdm import tqdm
import subprocess

import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from Bio.PDB import PDBParser
from Bio.PDB import Structure, Model, Chain, Residue, Atom


# import re
# def str2int(str):
#     return int(re.findall(r'\d+', str)[0])

def str2int(s):
    num = ""
    for i in s:
        if i.isdigit():
            num += i
    return int(num)


def pdb_chains(pdb_filepath, pdb_id=None):
    chains = {}
    parser = PDBParser()
    structure = parser.get_structure(pdb_id, pdb_filepath)
    for chain in structure.get_chains():
        positions = [res.id[1] for res in chain.get_residues()]
        chains[chain.id] = positions

    # for chain in chains:
    #     print(chain, chains[chain])
    return chains


def dms_matrix(df, name_col, value_col, chains):

    # 生成20种氨基酸的列表&肽链序号
    amino_acids = list("AVLIMFYWRHKDESTNQGCP")
    chains_matrix = {}
    for chain in chains:
        positions = chains[chain]
        chains_matrix[chain] = pd.DataFrame(index=positions, columns=amino_acids)

    # 解析name列，提取蛋白质序号、突变位点和目标氨基酸
    df['pdb_id'] = df[name_col].str.split('_').str[0]
    df['mutation'] = df[name_col].str.split('_').str[1]
    df['chain'] = df['mutation'].str[1]
    df['original_aa'] = df['mutation'].str[0]
    df['position'] = df['mutation'].apply(str2int)
    df['new_aa'] = df['mutation'].str[-1]

    # 填充heatmap_data
    for _, row in df.iterrows():
        chain = row['chain']
        chains_matrix[chain].at[row['position'], row['new_aa']] = row[value_col]

    # 将数据转换为数值类型
    for chain in chains_matrix:
        chains_matrix[chain] = chains_matrix[chain].astype(float)

    return chains_matrix


def draw_dms_matrix(chains_matrix):

    for chain in chains_matrix:
        dms_matrix = chains_matrix[chain]

        # 创建热图
        fig = px.imshow(
            dms_matrix.transpose(), 
            labels={'x': 'Amino Acid', 'y': 'Protein Position', 'color': 'y'},
            x=dms_matrix.index,
            y=dms_matrix.columns,
            aspect="auto",  # 保持正方形
            color_continuous_scale='RdBu'
        )

        # 设置每个数据点为正方形
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        # 设置布局
        fig.update_layout(
            title=f'Heatmap of Protein Mutations chain-{chain}',
            xaxis_title='Protein Position',
            yaxis_title='Amino Acid',
        )

        # 显示图表
        fig.show()



three_to_one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 
                'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 
                'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
amino_acids = list("AVLIMFYWRHKDESTNQGCP")

def DMS_in_silico(EvoEF2_toolpath, wt_pdb_filepath, pdb_id, mut_pdb_path):
    os.makedirs(mut_pdb_path, exist_ok=True)

    chains = {}
    parser = PDBParser()
    structure = parser.get_structure(pdb_id, wt_pdb_filepath)
    for chain in structure.get_chains():
        positions = [res.id[1] for res in chain.get_residues()]
        residues = [three_to_one[res.resname] for res in chain.get_residues()]
        assert len(positions) == len(residues)
        chains[chain.id] = {"chain_id": chain.id, "positions": positions, "residues": residues}

    df = pd.concat([pd.DataFrame(chains[chain]) for chain in chains], ignore_index=True)

    for i in tqdm(range(len(df))):
        print(f"Processing chain: {df['chain_id'][i]} pos: {df['positions'][i]}")
        mutation_pre = f"{df['residues'][i]}{df['chain_id'][i]}{df['positions'][i]}"

        mutations = [f"{mutation_pre}{aa}" for aa in amino_acids if aa != df['residues'][i]]
        for mutation in mutations:
            mut_name = f"{pdb_id}_{mutation}"
            print(f"{pdb_id} -> {mut_name}")

            with open(f"{mut_pdb_path}/{mut_name}.txt", 'w') as f:
                f.write(mutation + ';')
            
            cmd = f"{EvoEF2_toolpath} --command=BuildMutant \
                        --pdb={wt_pdb_filepath} --mutant_file={mut_pdb_path}/{mut_name}.txt && \
                    mv {pdb_id}_Model_0001.pdb {mut_pdb_path}/{mut_name}.pdb && \
                    rm {mut_pdb_path}/{mut_name}.txt"
            # print(cmd)
            subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL)



if __name__ == '__main__':

    EvoEF2_toolpath = "./tools/EvoEF2/EvoEF2"
    wt_pdb_filepath = "./datasets/skempi_v2/cleaned_PDBs/1AO7.pdb"
    pdb_id = "1AO7"
    mut_pdb_path = f"./datasets/DMS/{pdb_id}/mut_PDBs/"

    DMS_in_silico(EvoEF2_toolpath, wt_pdb_filepath, pdb_id, mut_pdb_path)


