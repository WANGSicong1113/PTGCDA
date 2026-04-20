import pandas as pd
import numpy as np
import math
import pickle
import os


# 生成高斯相互作用轮廓核相似度矩阵
def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def Getgauss_circRNA(adjacentmatrix, nc):
    """
    circRNA Gaussian interaction profile kernels similarity
    """
    KC = np.zeros((nc, nc))

    gamaa = 1
    sumnormc = 0
    for i in range(nc):
        normc = np.linalg.norm(adjacentmatrix[i]) ** 2
        sumnormc = sumnormc + normc
    gamac = gamaa / (sumnormc / nc)

    for i in range(nc):
        for j in range(nc):
            KC[i, j] = math.exp(
                -gamac * (np.linalg.norm(adjacentmatrix[i] - adjacentmatrix[j]) ** 2)
            )
    return KC


def Getgauss_disease(adjacentmatrix, nd):
    """
    Disease Gaussian interaction profile kernels similarity
    """
    KD = np.zeros((nd, nd))
    gamaa = 1
    sumnormd = 0
    for i in range(nd):
        normd = np.linalg.norm(adjacentmatrix[:, i]) ** 2
        sumnormd = sumnormd + normd
    gamad = gamaa / (sumnormd / nd)

    for i in range(nd):
        for j in range(nd):
            KD[i, j] = math.exp(
                -(
                    gamad
                    * (np.linalg.norm(adjacentmatrix[:, i] - adjacentmatrix[:, j]) ** 2)
                )
            )
    return KD


seed_everything(42)
adj_df = pd.read_csv(r"rd_adj.csv", index_col=0)
adj = adj_df.values
num_c, num_d = adj.shape

pos_ij = np.argwhere(adj == 1)
unlabelled_ij = np.argwhere(adj == 0)
np.random.shuffle(pos_ij)
np.random.shuffle(unlabelled_ij)
k_fold = 5
pos_ij_5fold = np.array_split(pos_ij, k_fold)
unlabelled_ij_5fold = np.array_split(unlabelled_ij, k_fold)

fold_cnt = 0

if (
    os.path.exists("pos_train_ij_list.pkl") and
    os.path.exists("pos_test_ij_list.pkl") and
    os.path.exists("unlabelled_train_ij_list.pkl") and
    os.path.exists("unlabelled_test_ij_list.pkl") and
    os.path.exists("c_gip_list.pkl") and
    os.path.exists("d_gip_list.pkl")):

    print("Lists Exists!")

    print("Loading pos_train_ij_list.pkl")
    with open('pos_train_ij_list.pkl', 'rb') as f:
        pos_train_ij_list = pickle.load(f)
    print("pos_train_ij_list.pkl Loaded")

    print("Loading pos_test_ij_list.pkl")
    with open('pos_test_ij_list.pkl', 'rb') as f:
        pos_test_ij_list = pickle.load(f)
    print("pos_test_ij_list.pkl Loaded")

    print("Loading unlabelled_train_ij_list")
    with open('unlabelled_train_ij_list.pkl', 'rb') as f:
        unlabelled_train_ij_list = pickle.load(f)
    print("unlabelled_train_ij_list Loaded")

    print("Loading unlabelled_test_ij_list")
    with open('unlabelled_test_ij_list.pkl', 'rb') as f:
        unlabelled_test_ij_list = pickle.load(f)
    print("unlabelled_test_ij_list Loaded")

    print("Loading unlabelled_test_ij_list")
    with open('c_gip_list.pkl', 'rb') as f:
        c_gip_list = pickle.load(f)
    print("unlabelled_test_ij_list Loaded")

    print("Loading unlabelled_test_ij_list")
    with open('d_gip_list.pkl', 'rb') as f:
        d_gip_list = pickle.load(f)
    print("unlabelled_test_ij_list Loaded")
else:
    print("Lists Not Exists!")
    print("Creating List")

    pos_train_ij_list = []
    pos_test_ij_list = []
    unlabelled_train_ij_list = []
    unlabelled_test_ij_list = []
    c_gip_list = []
    d_gip_list = []

    for i in range(k_fold):
        extract_idx = list(range(k_fold))
        extract_idx.remove(i)

        pos_train_ij = np.vstack([pos_ij_5fold[idx] for idx in extract_idx])
        pos_test_ij = pos_ij_5fold[i]

        unlabelled_train_ij = np.vstack([unlabelled_ij_5fold[idx] for idx in extract_idx])
        unlabelled_test_ij = unlabelled_ij_5fold[i]

        A = np.zeros_like(adj)
        A[tuple(list(pos_train_ij.T))] = 1

        c_gip = Getgauss_circRNA(A, num_c)
        d_gip = Getgauss_disease(A, num_d)

        pos_train_ij_list.append(pos_train_ij)
        pos_test_ij_list.append(pos_test_ij)
        unlabelled_train_ij_list.append(unlabelled_train_ij)
        unlabelled_test_ij_list.append(unlabelled_test_ij)
        c_gip_list.append(c_gip)
        d_gip_list.append(d_gip)

        fold_cnt = fold_cnt + 1

    with open("pos_train_ij_list.pkl", "wb") as f:
        pickle.dump(pos_train_ij_list, f)

    with open("pos_test_ij_list.pkl", "wb") as f:
        pickle.dump(pos_test_ij_list, f)

    with open("unlabelled_train_ij_list.pkl", "wb") as f:
        pickle.dump(unlabelled_train_ij_list, f)

    with open("unlabelled_test_ij_list.pkl", "wb") as f:
        pickle.dump(unlabelled_test_ij_list, f)

    with open("c_gip_list.pkl", "wb") as f:
        pickle.dump(c_gip_list, f)

    with open("d_gip_list.pkl", "wb") as f:
        pickle.dump(d_gip_list, f)

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# 加载RNA相似性矩阵
rna_sim = pd.read_csv('cosine_similarity_matrix.csv')

# 加载疾病相似性矩阵
disease_sim = pd.read_csv('d2d_do.csv')

# 加载RNA和疾病相关性矩阵
adj_matrix = pd.read_csv('rd_adj.csv')

# 加载GIP相似性数据
with open('fold_info.pickle', 'rb') as f:
    fold_info = pickle.load(f)
    c_gip_list = fold_info['c_gip_list']
    d_gip_list = fold_info['d_gip_list']
