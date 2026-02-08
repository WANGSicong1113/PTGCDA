import os
import numpy as np
from GIP import *
from model import *
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import pickle
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib
import matplotlib.pyplot as plt
from utils import *
device = torch.device("cpu")
seed_everything(42)
path = "./scores/"
if not os.path.exists(path):
    os.makedirs(path)
adj_np = pd.read_csv(r"rd_adj.csv", index_col=0).values
c_sim_df = pd.read_csv(r"cosine_similarity_matrix.csv", index_col=0)
c_sim_np = c_sim_df.values
gensim_feat = np.load(
    r"gensim_feat_128.npy",
    allow_pickle=True,
).flat[0]
c_kmers_emb = gensim_feat["p_kmers_emb"]
pad_c_kmers_id_seq = gensim_feat["pad_kmers_id_seq"]
d_sim_np = pd.read_csv(r"d2d_do.csv", index_col=0).values
d_feat = d_sim_np
num_c, num_d = adj_np.shape
c_sim = torch.FloatTensor(c_sim_np).to(device)
d_sim = torch.FloatTensor(d_sim_np).to(device)
adj = torch.FloatTensor(adj_np).to(device)
c_kmers_emb = torch.FloatTensor(c_kmers_emb).to(device)
pad_c_kmers_id_seq = torch.tensor(pad_c_kmers_id_seq).to(device)
d_feat = torch.FloatTensor(d_feat).to(device)
k = 1
merge_win_size = 32
context_size_list = [1, 3, 5]
dll_out_size = 128 * len(context_size_list) * k
graph_out_dim = 128 * k
graph_hidden_dim = 128 * k
num_layers, dropout = 1, 0.6
query_size = key_size = 128 * k
value_size = 128 * k
enc_ffn_num_hiddens, n_enc_heads = 128, 2 * k
lr, num_epochs = 0.006, 200
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
    pad_c_kmers_id_seq,
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
        import numpy as np
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
    for epoch in range(num_epochs):
        if epoch < warmup_epochs:
            warmup_lr = lr * (warmup_factor + (1 - warmup_factor) * (epoch + 1) / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        model.train()
        optimizer.zero_grad()
        pred = model(pad_c_kmers_id_seq, d_feat, adj_full)
        train_loss, test_loss = loss(pred, adj, train_mask, test_mask)
        if use_pairwise and pos_train_ij is not None and rn_ij is not None:
            pw_loss = pairwise_auc_loss(pred, pos_train_ij, rn_ij, sample_pairs=num_pairs)
            total_loss = (1 - pairwise_weight) * train_loss + pairwise_weight * pw_loss
        else:
            total_loss = train_loss
        total_loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            pred = model(pad_c_kmers_id_seq, d_feat, adj_full)
        scores = pred[tuple(list(test_idx.T))].cpu().detach().numpy()
        np.save(rf"./scores/f{fold_cnt}_e{epoch}_scores.npy", scores)
        from sklearn.metrics import roc_auc_score
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
        best_model_dir = "./best_models"
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)
        best_model_path = os.path.join(best_model_dir, f"best_model_fold{fold_cnt}.pth")
        torch.save(best_model_state, best_model_path)


logger = Logger(5)
with open(r"fold_info.pickle", "rb") as f:
    fold_info = pickle.load(f)
with open("rn_ij_list_pu.pickle", "rb") as f:
    rn_ij_list_pu = pickle.load(f)
with open(rf"rn_ij_list_spy_fast.pickle", "rb") as f:
    rn_ij_list_spy = pickle.load(f)
with open(rf"rn_ij_list_two.pickle","rb") as f :
    rn_ij_list_two = pickle.load(f)
pos_train_ij_list = fold_info["pos_train_ij_list"]
pos_test_ij_list = fold_info["pos_test_ij_list"]
unlabelled_train_ij_list = fold_info["unlabelled_train_ij_list"]
unlabelled_test_ij_list = fold_info["unlabelled_test_ij_list"]
c_gip_list = fold_info["c_gip_list"]
d_gip_list = fold_info["d_gip_list"]
for i in range(5):
    pos_train_ij = pos_train_ij_list[i]
    pos_test_ij = pos_test_ij_list[i]
    unlabelled_train_ij = unlabelled_train_ij_list[i]
    unlabelled_test_ij = unlabelled_test_ij_list[i]
    c_gip = c_gip_list[i]
    d_gip = d_gip_list[i]
    rn_ij = rn_ij_list_two[i]
    A_corner_np = np.zeros_like(adj_np)
    A_corner_np[tuple(list(pos_train_ij.T))] = 1
    A_np = np.concatenate(
        (
            np.concatenate(((c_sim_np + c_gip) / 2, A_corner_np), axis=1),
            np.concatenate(((A_corner_np).T, (d_sim_np + d_gip) / 2), axis=1),
        ),
        axis=0,
    )
    train_mask_np = np.zeros_like(adj_np)
    train_mask_np[tuple(list(pos_train_ij.T))] = 1
    train_mask_np[tuple(list(rn_ij.T))] = 1
    test_mask_np = np.zeros_like(adj_np)
    test_mask_np[tuple(list(pos_test_ij.T))] = 1
    test_mask_np[tuple(list(unlabelled_test_ij.T))] = 1
    A_corner = torch.FloatTensor(A_corner_np).to(device)
    A = torch.FloatTensor(A_np).to(device)
    train_mask = torch.FloatTensor(train_mask_np).to(device)
    test_mask = torch.FloatTensor(test_mask_np).to(device)
    torch.cuda.empty_cache()
    deep_lnc_loc = DeepLncLoc(
        c_kmers_emb, dropout, merge_win_size, context_size_list, dll_out_size
    ).to(device)
    graph_sage = GraphSAGE(
        p_feat_dim=dll_out_size,
        d_feat_dim=feat_init_d,
        n_hidden=graph_hidden_dim,
        dropout=dropout,
    ).to(device)
    c_encoder = TransformerEncoder(
        q_in_dim=graph_out_dim,
        kv_in_dim=graph_out_dim,
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
        q_in_dim=graph_out_dim,
        kv_in_dim=graph_out_dim,
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
    model = PTGCDA(deep_lnc_loc, graph_sage, c_encoder, d_encoder, predictor).to(device)
    fit(
        i,
        model,
        adj,
        A,
        pad_c_kmers_id_seq,
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
logger.save("circRNA_result")
