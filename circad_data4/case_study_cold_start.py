"""
Strict cold-start case study for PTGCDA.

Training omits all known associations involving the target diseases; validation
uses held-out positives and sampled unknown negatives for those diseases only.
Data loading, splits, and graph construction follow ``case_study.ipynb``, except
that the adjacency used for GIP and message passing is restricted to non-target
edges.

- Row/column labels of ``rd_adj.csv`` define circRNA and disease identifiers.
- Gaussian interaction profile (GIP) kernels are computed from non-target
  associations only.
"""
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from GIP import Getgauss_circRNA, Getgauss_disease
from model import (
    DeepLncLoc,
    GraphSAGE,
    PTGCDA,
    Predictor,
    TransformerEncoder,
)
from utils import seed_everything


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_basic_data(device):
    """Load adjacency and features from rd_adj.csv (same as case_study.ipynb)."""
    adj_df = pd.read_csv("rd_adj.csv", index_col=0)
    adj_np = adj_df.values
    circrna_names = adj_df.index.tolist()
    disease_names = adj_df.columns.tolist()

    p_sim_df = pd.read_csv("cosine_similarity_matrix.csv", index_col=0)
    p_sim_np = p_sim_df.values

    d_sim_np = pd.read_csv("d2d_do.csv", index_col=0).values

    gensim_feat = np.load("gensim_feat_128.npy", allow_pickle=True).flat[0]
    p_kmers_emb = gensim_feat["p_kmers_emb"]
    pad_kmers_id_seq = gensim_feat["pad_kmers_id_seq"]

    d_feat = d_sim_np
    num_c, num_d = adj_np.shape

    adj = torch.FloatTensor(adj_np).to(device)
    p_sim = torch.FloatTensor(p_sim_np).to(device)
    d_sim = torch.FloatTensor(d_sim_np).to(device)
    p_kmers_emb = torch.FloatTensor(p_kmers_emb).to(device)
    pad_kmers_id_seq = torch.tensor(pad_kmers_id_seq).to(device)
    d_feat = torch.FloatTensor(d_feat).to(device)

    return (
        adj_np,
        p_sim_np,
        d_sim_np,
        adj,
        p_sim,
        d_sim,
        d_feat,
        p_kmers_emb,
        pad_kmers_id_seq,
        num_c,
        num_d,
        circrna_names,
        disease_names,
    )


def load_fold_and_neg():
    """fold_info + rn_ij_list_two (same as case_study.ipynb)."""
    with open("fold_info.pickle", "rb") as f:
        fold_info = pickle.load(f)
    with open("rn_ij_list_two.pickle", "rb") as f:
        rn_ij_list_two = pickle.load(f)
    return fold_info, rn_ij_list_two


def build_all_pos_and_rn(fold_info, rn_ij_list_two):
    """Merge positive and negative pairs from all folds (same as case_study.ipynb)."""
    all_pos_ij = []
    for fold_idx in range(len(fold_info["pos_train_ij_list"])):
        all_pos_ij.extend(fold_info["pos_train_ij_list"][fold_idx])
        all_pos_ij.extend(fold_info["pos_test_ij_list"][fold_idx])
    all_pos_ij = np.array(all_pos_ij)

    all_rn_ij = []
    for fold_idx in range(len(rn_ij_list_two)):
        all_rn_ij.extend(rn_ij_list_two[fold_idx])
    all_rn_ij = (
        np.array(all_rn_ij)
        if len(all_rn_ij) > 0
        else np.array([]).reshape(0, 2)
    )
    return all_pos_ij, all_rn_ij


def get_target_disease_indices(target_diseases, disease_names):
    """Map target disease names to column indices in rd_adj."""
    target_disease_indices = []
    found = []
    for name in target_diseases:
        if name in disease_names:
            idx = disease_names.index(name)
            target_disease_indices.append(idx)
            found.append(name)
        else:
            matches = [
                d
                for d in disease_names
                if name.lower() in d.lower() or d.lower() in name.lower()
            ]
            raise ValueError(
                f"Disease not found: '{name}'. Possible matches in rd_adj columns: {matches[:5]}"
            )
    return target_disease_indices, found


def build_cold_start_splits(all_pos_ij, all_rn_ij, target_disease_indices, adj_np):
    """
    Cold-start split (strict):
    - Train positives: all known edges except those for target diseases
    - Train negatives: sampled from PU negatives not involving target diseases (ratio 0.71, as in case_study)
    - Val positives: all known edges for target diseases
    - Val negatives: same count sampled from unknown edges for target diseases
    """
    num_c, num_d = adj_np.shape
    target_set = set(target_disease_indices)

    # Train positives: non-target diseases only
    remaining_mask = np.array(
        [row[1] not in target_set for row in all_pos_ij]
    )
    train_pos_ij = all_pos_ij[remaining_mask]
    target_pos_ij = all_pos_ij[~remaining_mask]

    if len(target_pos_ij) == 0:
        raise ValueError(
            "Target diseases have no known associations; cold-start validation is impossible."
        )

    # Train negatives: from non-target PU negatives, subsample to 0.71 ratio (case_study)
    remaining_rn = []
    for row in all_rn_ij:
        if row[1] not in target_set:
            remaining_rn.append(row)
    remaining_rn_ij = np.array(remaining_rn) if remaining_rn else np.array([]).reshape(0, 2)

    target_neg_ratio = 0.71
    target_neg_count = int(len(train_pos_ij) * target_neg_ratio)
    if len(remaining_rn_ij) > target_neg_count:
        np.random.seed(42)
        sampled = np.random.choice(
            len(remaining_rn_ij), size=target_neg_count, replace=False
        )
        train_neg_ij = remaining_rn_ij[sampled]
    else:
        train_neg_ij = remaining_rn_ij

    # Validation: target positives + same number of unknown negatives
    unknown_list = []
    for p in range(num_c):
        for d in target_disease_indices:
            if adj_np[p, d] == 0:
                unknown_list.append([p, d])
    unknown_ij = np.array(unknown_list) if unknown_list else np.array([]).reshape(0, 2)
    np.random.seed(42)
    if len(unknown_ij) > 0:
        np.random.shuffle(unknown_ij)
    n_val_pos = len(target_pos_ij)
    val_neg_ij = unknown_ij[:n_val_pos] if len(unknown_ij) >= n_val_pos else unknown_ij

    return train_pos_ij, train_neg_ij, target_pos_ij, val_neg_ij


class MaskedBCELoss(nn.BCELoss):
    def forward(self, pred, adj, train_mask, test_mask):
        self.reduction = "none"
        unweighted_loss = super(MaskedBCELoss, self).forward(pred, adj)
        train_loss = (unweighted_loss * train_mask).sum()
        test_loss = (unweighted_loss * test_mask).sum()
        return train_loss, test_loss


def evaluate_auc_aupr(scores, adj_np, val_pos_ij, val_neg_ij):
    """AUC / AUPR on validation (known target edges vs sampled unknown)."""
    from sklearn import metrics

    idx = np.vstack([val_pos_ij, val_neg_ij])
    if len(idx) == 0:
        return None, None
    labels = np.hstack(
        [np.ones(len(val_pos_ij)), np.zeros(len(val_neg_ij))]
    ).astype(np.float64)
    r, c = torch.LongTensor(idx[:, 0]).to(scores.device), torch.LongTensor(idx[:, 1]).to(scores.device)
    scr = scores[r, c].cpu().detach().numpy()

    fpr, tpr, _ = metrics.roc_curve(labels, scr)
    auc = metrics.auc(fpr, tpr)
    precisions, recalls, _ = metrics.precision_recall_curve(labels, scr)
    aupr = metrics.auc(recalls, precisions)
    return float(auc), float(aupr)


def main():
    seed_everything(42)
    device = get_device()
    print(f"Device: {device}")

    TARGET_DISEASES = [
        "lung cancer",
        "bladder cancer",
        "breast cancer",
        "colorectal cancer",
        "gastric cancer",  # column name in rd_adj is lowercase
        "hepatocellular carcinoma",
        "pancreatic cancer",
    ]
    print("Target diseases (cold start, Top-10 each):", TARGET_DISEASES)

    (
        adj_np,
        p_sim_np,
        d_sim_np,
        adj,
        p_sim,
        d_sim,
        d_feat,
        p_kmers_emb,
        pad_kmers_id_seq,
        num_c,
        num_d,
        circrna_names,
        disease_names,
    ) = load_basic_data(device)

    fold_info, rn_ij_list_two = load_fold_and_neg()
    all_pos_ij, all_rn_ij = build_all_pos_and_rn(fold_info, rn_ij_list_two)

    target_disease_indices, found = get_target_disease_indices(
        TARGET_DISEASES, disease_names
    )
    print("Target disease indices:", target_disease_indices, "names:", found)

    train_pos_ij, train_neg_ij, val_pos_ij, val_neg_ij = build_cold_start_splits(
        all_pos_ij, all_rn_ij, target_disease_indices, adj_np
    )

    print("Cold-start split:")
    print("  Train positives (non-target):", len(train_pos_ij))
    print("  Train negatives (after sampling):", len(train_neg_ij))
    print("  Val positives (target, known):", len(val_pos_ij))
    print("  Val negatives (target, unknown):", len(val_neg_ij))

    # Graph: GIP from non-target associations only (strict cold start)
    adj_np_for_graph = np.zeros_like(adj_np)
    if len(train_pos_ij) > 0:
        adj_np_for_graph[train_pos_ij[:, 0], train_pos_ij[:, 1]] = 1

    p_gip = Getgauss_circRNA(adj_np_for_graph, num_c)
    d_gip = Getgauss_disease(adj_np_for_graph.T, num_d)

    A_corner_np = np.zeros_like(adj_np)
    if len(train_pos_ij) > 0:
        A_corner_np[train_pos_ij[:, 0], train_pos_ij[:, 1]] = 1

    A_np = np.concatenate(
        (
            np.concatenate(((p_sim_np + p_gip) / 2, A_corner_np), axis=1),
            np.concatenate((A_corner_np.T, (d_sim_np + d_gip) / 2), axis=1),
        ),
        axis=0,
    )
    A = torch.FloatTensor(A_np).to(device)

    # Train / val masks (full matrix, same as main / case_study)
    train_mask_np = np.zeros_like(adj_np, dtype=float)
    train_mask_np[train_pos_ij[:, 0], train_pos_ij[:, 1]] = 1.0
    if len(train_neg_ij) > 0:
        train_mask_np[train_neg_ij[:, 0], train_neg_ij[:, 1]] = 1.0

    test_mask_np = np.zeros_like(adj_np, dtype=float)
    test_mask_np[val_pos_ij[:, 0], val_pos_ij[:, 1]] = 1.0
    if len(val_neg_ij) > 0:
        test_mask_np[val_neg_ij[:, 0], val_neg_ij[:, 1]] = 1.0

    train_mask = torch.FloatTensor(train_mask_np).to(device)
    test_mask = torch.FloatTensor(test_mask_np).to(device)

    # Hyperparameters (aligned with case_study / main)
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

    # Multiple runs; average logits for final ranking
    N_RUNS = 5
    logits_sum = torch.zeros((num_c, num_d), device=device)
    best_aucs = []

    print(
        f"Starting {N_RUNS} cold-start runs; Top-10 CSV uses mean logits across runs."
    )

    for run in range(N_RUNS):
        print(f"\n==== Cold-start run {run + 1}/{N_RUNS} ====")
        seed_everything(42 + run)

        deep_lnc_loc = DeepLncLoc(
            p_kmers_emb, dropout, merge_win_size, context_size_list, dll_out_size
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
        model = PTGCDA(deep_lnc_loc, graph_sage, c_encoder, d_encoder, predictor).to(
            device
        )

        loss_fn = MaskedBCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        warmup_epochs = 10
        warmup_factor = 0.1
        best_auc = 0.0
        best_state = None

        for epoch in range(num_epochs):
            if epoch < warmup_epochs:
                warmup_lr = lr * (
                    warmup_factor
                    + (1 - warmup_factor) * (epoch + 1) / warmup_epochs
                )
                for g in optimizer.param_groups:
                    g["lr"] = warmup_lr

            model.train()
            optimizer.zero_grad()
            pred = model(pad_kmers_id_seq, d_feat, A)
            train_loss, _ = loss_fn(pred, adj, train_mask, test_mask)
            train_loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                scores = model(pad_kmers_id_seq, d_feat, A)
            auc, aupr = evaluate_auc_aupr(scores, adj_np, val_pos_ij, val_neg_ij)

            print(
                f"[Run {run+1}] Epoch {epoch:03d} | train_loss={train_loss.item():.4f} "
                f"| cold-start AUC={auc:.4f} | AUPR={aupr:.4f}"
            )

            if auc is not None and auc > best_auc:
                best_auc = auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        best_aucs.append(best_auc)

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            p_feat_dll = deep_lnc_loc(pad_kmers_id_seq)
            p_feat_gcn, d_feat_gcn = graph_sage(p_feat_dll, d_feat, A)
            p_enc = c_encoder(p_feat_gcn, d_feat_gcn)
            d_enc = d_encoder(d_feat_gcn, p_feat_gcn)
            logits_run = p_enc.mm(d_enc.t())

        logits_sum += logits_run

    logits_full = logits_sum / float(N_RUNS)
    if len(best_aucs) > 0:
        print(
            f"\nCold-start runs finished: mean AUC = {np.mean(best_aucs):.4f} "
            f"(std = {np.std(best_aucs):.4f})"
        )

    # Top-10 unknown pairs per target disease (pre-sigmoid logits)
    rows = []
    for d_idx in target_disease_indices:
        disease_name = disease_names[d_idx]
        cands = []
        for p_idx in range(num_c):
            if adj_np[p_idx, d_idx] == 0:
                cands.append((p_idx, logits_full[p_idx, d_idx].item()))
        cands.sort(key=lambda x: x[1], reverse=True)
        for p_idx, logit in cands[:10]:
            rows.append({
                "Disease": disease_name,
                "circRNA": circrna_names[p_idx],
                "circRNA_index": p_idx,
                "Logit_Score": logit,
            })

    df_top = pd.DataFrame(rows)
    out_path = "case_study_cold_start_top10_predictions.csv"
    df_top.to_csv(out_path, index=False)
    print(f"Saved Top-10 predictions: {out_path}")


if __name__ == "__main__":
    main()
