import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import roc_auc_score

from model import (
    DeepLncLoc,
    GraphSAGE,
    PTGCDA,
    Predictor,
    TransformerEncoder,
)
from utils import Logger, seed_everything

DATA_DIR = os.path.abspath(os.environ.get("CIRCAD_DATA_DIR", os.getcwd()))

SCORES_DIR = os.path.join(DATA_DIR, "scores")
BEST_MODELS_DIR = os.path.join(DATA_DIR, "best_models")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

os.makedirs(SCORES_DIR, exist_ok=True)
os.makedirs(BEST_MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cpu")
seed_everything(42)

# load adj, sim
# Load adjacent matrix (data_processing.py)
adj_np = pd.read_csv(os.path.join(DATA_DIR, "rd_adj.csv"), index_col=0).values
# Load circRNA similarity (gen_half_p2p_simth.py)
p_sim_df = pd.read_csv(os.path.join(DATA_DIR, "cosine_similarity_matrix.csv"), index_col=0)
p_sim_np = p_sim_df.values
# Load circRNA feature (gen_pfeat_gensim.py)
gensim_feat = np.load(
    os.path.join(DATA_DIR, "gensim_feat_128.npy"),
    allow_pickle=True,
).flat[0]
p_kmers_emb = gensim_feat["p_kmers_emb"]
pad_kmers_id_seq = gensim_feat["pad_kmers_id_seq"]
# Load disease similarity (gen_d2d_do.py)
d_sim_np = pd.read_csv(os.path.join(DATA_DIR, "d2d_do.csv"), index_col=0).values
d_feat = d_sim_np

num_c, num_d = adj_np.shape


p_sim = torch.FloatTensor(p_sim_np).to(device)
d_sim = torch.FloatTensor(d_sim_np).to(device)
adj = torch.FloatTensor(adj_np).to(device)
p_kmers_emb = torch.FloatTensor(p_kmers_emb).to(device)
pad_kmers_id_seq = torch.tensor(pad_kmers_id_seq).to(device)
d_feat = torch.FloatTensor(d_feat).to(device)

k = 1
merge_win_size = 32
context_size_list = [1, 3, 5]
dll_out_size = 128 * len(context_size_list) * k

gcn_out_dim = 128 * k
gcn_hidden_dim = 128 * k
num_layers, dropout = 1, 0.6
query_size = key_size = 128 * k
value_size = 128 * k
enc_ffn_num_hiddens, n_enc_heads = 128, 2 * k

lr, num_epochs = 0.006, 500
feat_init_d = d_feat.shape[1]


class MaskedBCELoss(nn.BCELoss):
    def forward(self, pred, adj, train_mask, test_mask):
        self.reduction = "none"
        unweighted_loss = super(MaskedBCELoss, self).forward(pred, adj)
        train_loss = (unweighted_loss * train_mask).sum()
        test_loss = (unweighted_loss * test_mask).sum()
        return train_loss, test_loss


def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad and p.grad is not None]
    else:
        params = [p for p in net.params if p.grad is not None]
    if len(params) == 0:
        return
    norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def fit(
    fold_cnt,
    model,
    adj,
    adj_full,
    pad_kmers_id_seq,
    d_feat,
    train_mask,
    test_mask,
    lr,
    num_epochs,
    pos_train_ij=None,
    rn_ij=None,
    use_pairwise=False,
    pairwise_weight=0.5,
    num_pairs=20000,
):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

    model.apply(xavier_init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    warmup_epochs = 10
    warmup_factor = 0.1
    
    best_auc = 0.0
    best_auc_epoch = 0
    best_model_state = None
    
    auc_drop_threshold = 0.001
    consecutive_drops = 0
    max_consecutive_drops = 1
    
    loss = MaskedBCELoss()

    def pairwise_auc_loss(scores, pos_idx_np, neg_idx_np, sample_pairs=20000):
        if len(pos_idx_np) == 0 or len(neg_idx_np) == 0:
            return torch.tensor(0.0, device=scores.device)
        n_pos = pos_idx_np.shape[0]
        n_neg = neg_idx_np.shape[0]
        sp = min(sample_pairs, n_pos * n_neg)
        pos_sel = np.random.randint(0, n_pos, size=sp)
        neg_sel = np.random.randint(0, n_neg, size=sp)
        pos_pairs = pos_idx_np[pos_sel]
        neg_pairs = neg_idx_np[neg_sel]
        s_pos = scores[tuple(pos_pairs.T)]
        s_neg = scores[tuple(neg_pairs.T)]
        return -torch.nn.functional.logsigmoid(s_pos - s_neg).mean()

    test_idx = torch.argwhere(test_mask == 1)
    # test_idx = torch.argwhere(torch.ones_like(test_mask) == 1)
    
    for epoch in range(num_epochs):
        if epoch < warmup_epochs:
            warmup_lr = lr * (warmup_factor + (1 - warmup_factor) * (epoch + 1) / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        model.train()
        optimizer.zero_grad()
        pred = model(pad_kmers_id_seq, d_feat, adj_full)
        train_loss, test_loss = loss(pred, adj, train_mask, test_mask)

        if use_pairwise and pos_train_ij is not None and rn_ij is not None:
            pw_loss = pairwise_auc_loss(pred, pos_train_ij, rn_ij, sample_pairs=num_pairs)
            total_loss = (1 - pairwise_weight) * train_loss + pairwise_weight * pw_loss
        else:
            total_loss = train_loss

        total_loss.backward()
        
        # grad_clipping(model, 0.5)
        
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(pad_kmers_id_seq, d_feat, adj_full)

        scores = pred[tuple(list(test_idx.T))].cpu().detach().numpy()
        np.save(os.path.join(SCORES_DIR, f"f{fold_cnt}_e{epoch}_scores.npy"), scores)

        labels = adj[tuple(list(test_idx.T))].cpu().detach().numpy()
        current_auc = roc_auc_score(labels, scores)
        
        logger.update(
            fold_cnt, epoch, adj, pred, test_idx, train_loss.item(), test_loss.item()
        )
        
        if current_auc > best_auc:
            best_auc = current_auc
            best_auc_epoch = epoch
            consecutive_drops = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            if epoch >= warmup_epochs:
                auc_drop = best_auc - current_auc
                if auc_drop > auc_drop_threshold:
                    consecutive_drops += 1
                    if consecutive_drops >= max_consecutive_drops:
                        if best_model_state is not None:
                            model.load_state_dict(best_model_state)
                            for param_group in optimizer.param_groups:
                                old_lr = param_group['lr']
                                param_group['lr'] = old_lr * 0.9
                        consecutive_drops = 0
                else:
                    consecutive_drops = 0
        
    if best_model_state is not None:
        best_model_dir = BEST_MODELS_DIR
        best_model_path = os.path.join(best_model_dir, f"best_model_fold{fold_cnt}.pth")
        torch.save(best_model_state, best_model_path)
    else:
        pass

logger = Logger(5)

with open(os.path.join(DATA_DIR, "fold_info.pickle"), "rb") as f:
    fold_info = pickle.load(f)
with open(os.path.join(DATA_DIR, "rn_ij_list_pu.pickle"), "rb") as f:
    rn_ij_list_pu = pickle.load(f)
with open(os.path.join(DATA_DIR, "rn_ij_list_spy_fast.pickle"), "rb") as f:
    rn_ij_list_spy = pickle.load(f)
with open(os.path.join(DATA_DIR, "rn_ij_list_two.pickle"), "rb") as f:
    rn_ij_list_two = pickle.load(f)


pos_train_ij_list = fold_info["pos_train_ij_list"]
pos_test_ij_list = fold_info["pos_test_ij_list"]
unlabelled_train_ij_list = fold_info["unlabelled_train_ij_list"]
unlabelled_test_ij_list = fold_info["unlabelled_test_ij_list"]
p_gip_list = fold_info["c_gip_list"]
d_gip_list = fold_info["d_gip_list"]
for i in range(5):
    pos_train_ij = pos_train_ij_list[i]
    pos_test_ij = pos_test_ij_list[i]
    unlabelled_train_ij = unlabelled_train_ij_list[i]
    unlabelled_test_ij = unlabelled_test_ij_list[i]
    p_gip = p_gip_list[i]
    d_gip = d_gip_list[i]
    # rn_ij = np.concatenate((rn_ij_list_two[i], rn_ij_list_spy[i]))
    # rn_ij = rn_ij_list_spy[i]
    rn_ij = rn_ij_list_two[i]
    # rn_ij = rn_ij_list_pu[i]
    A_corner_np = np.zeros_like(adj_np)
    A_corner_np[tuple(list(pos_train_ij.T))] = 1

    A_np = np.concatenate(
        (
            np.concatenate(((p_sim_np + p_gip) / 2, A_corner_np), axis=1),
            np.concatenate(((A_corner_np).T, (d_sim_np + d_gip) / 2), axis=1),
        ),
        axis=0,
    )

    # train_mask_np = np.ones_like(adj_np)
    train_mask_np = np.zeros_like(adj_np)
    train_mask_np[tuple(list(pos_train_ij.T))] = 1
    # train_mask_np[tuple(list(unlabelled_train_ij.T))] = 1
    train_mask_np[tuple(list(rn_ij.T))] = 1

    test_mask_np = np.zeros_like(adj_np)
    test_mask_np[tuple(list(pos_test_ij.T))] = 1
    test_mask_np[tuple(list(unlabelled_test_ij.T))] = 1
    num_positive_samples = np.sum(train_mask_np == 1)
    num_negative_samples = np.sum(train_mask_np == 0)

    A_corner = torch.FloatTensor(A_corner_np).to(device)
    A = torch.FloatTensor(A_np).to(device)
    train_mask = torch.FloatTensor(train_mask_np).to(device)
    test_mask = torch.FloatTensor(test_mask_np).to(device)

    torch.cuda.empty_cache()
    deep_lnc_loc = DeepLncLoc(
        p_kmers_emb, dropout, merge_win_size, context_size_list, dll_out_size
    ).to(device)

    graph_sage = GraphSAGE(
        p_feat_dim=dll_out_size,
        d_feat_dim=feat_init_d,
        n_hidden=gcn_hidden_dim,
        dropout=dropout,
    ).to(device)

    p_encoder = TransformerEncoder(
        q_in_dim=gcn_out_dim,
        kv_in_dim=gcn_out_dim,
        key_size=key_size,
        query_size=query_size,
        value_size=value_size,
        ffn_num_hiddens=enc_ffn_num_hiddens,
        num_heads=n_enc_heads,
        num_layers=num_layers,
        dropout=dropout,
        bias=False,
    ).to(device)

    d_encoder = TransformerEncoder(
        q_in_dim=gcn_out_dim,
        kv_in_dim=gcn_out_dim,
        key_size=key_size,
        query_size=query_size,
        value_size=value_size,
        ffn_num_hiddens=enc_ffn_num_hiddens,
        num_heads=n_enc_heads,
        num_layers=num_layers,
        dropout=dropout,
        bias=False,
    ).to(device)

    predictor = Predictor().to(device)

    use_pairwise_flag = True
    pairwise_weight = 0.2
    model = PTGCDA(deep_lnc_loc, graph_sage, p_encoder, d_encoder, predictor).to(device)
    fit(
        i,
        model,
        adj,
        A,
        pad_kmers_id_seq,
        d_feat,
        train_mask,
        test_mask,
        lr,
        num_epochs,
        pos_train_ij=pos_train_ij,
        rn_ij=rn_ij,
        use_pairwise=use_pairwise_flag,
        pairwise_weight=pairwise_weight,
        num_pairs=20000,
    )
    max_allocated_memory = torch.cuda.max_memory_allocated()
    logger.save(os.path.join(RESULTS_DIR, "circRNA_result"))
# torch.save(model, os.path.join(DATA_DIR, "params.pt"))
