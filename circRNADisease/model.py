import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class DeepLncLoc(nn.Module):
    """DeepLncLoc: Sequence feature extraction module using CNN."""
    def __init__(self, w2v_emb, dropout, merge_win_size, context_size_list, out_size):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(w2v_emb, freeze=False)
        self.dropout = nn.Dropout(dropout)
        self.merge_win_size = merge_win_size
        # 使用 AdaptiveAvgPool1d，但在 forward 中处理 MPS 兼容性
        self.merge_win = nn.AdaptiveAvgPool1d(merge_win_size)
        assert out_size % len(context_size_list) == 0
        filter_out_size = int(out_size / len(context_size_list))
        self.con_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=w2v_emb.shape[1],
                        out_channels=filter_out_size,
                        kernel_size=context_size_list[i],
                    ),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d(1),
                )
                for i in range(len(context_size_list))
            ]
        )

    def forward(self, p_kmers_id):
        # p_kmers: num_p × all kmers of each p
        x = self.dropout(self.embedding(p_kmers_id))
        x = x.transpose(1, 2)  # [batch, channels, length]
        
        # 处理 MPS 兼容性：确保输入长度能被输出长度整除
        if x.device.type == "mps":
            seq_len = x.shape[2]
            if seq_len % self.merge_win_size != 0:
                # 如果不可整除，使用普通池化替代
                kernel_size = seq_len // self.merge_win_size
                if kernel_size == 0:
                    kernel_size = 1
                stride = kernel_size
                x = F.avg_pool1d(x, kernel_size=kernel_size, stride=stride)
                # 如果还有余数，进行裁剪或填充到目标长度
                if x.shape[2] != self.merge_win_size:
                    if x.shape[2] > self.merge_win_size:
                        x = x[:, :, :self.merge_win_size]
                    else:
                        # 填充到目标长度
                        pad_size = self.merge_win_size - x.shape[2]
                        x = F.pad(x, (0, pad_size), mode='constant', value=0)
            else:
                x = self.merge_win(x)
        else:
            x = self.merge_win(x)
        
        x = [conv(x).squeeze(dim=2) for conv in self.con_list]
        x = torch.cat(x, dim=1)
        return x


class AddNorm(nn.Module):
    """Add & Norm layer for Transformer."""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


def transpose_qkv(X, num_heads):
    """Transpose for multi-head attention computation."""
    X = X.reshape(X.shape[0], num_heads, -1)
    X = X.permute(1, 0, 2)
    return X


def transpose_output(X, num_heads):
    """Transpose output back to original shape."""
    X = X.permute(1, 0, 2)
    return X.reshape(X.shape[0], -1)


class DotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q_p, K_p, V_p):
        d = Q_p.shape[-1]
        scores = torch.bmm(Q_p, K_p.transpose(1, 2)) / np.sqrt(d)
        self.attention_weights = nn.functional.softmax(scores, dim=-1)
        return torch.bmm(self.dropout(self.attention_weights), V_p)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    def __init__(
        self,
        q_in_dim,
        kv_in_dim,
        key_size,
        query_size,
        value_size,
        num_heads,
        dropout,
        bias=False,
        **kwargs
    ):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(q_in_dim, query_size, bias=bias)
        self.W_k = nn.Linear(kv_in_dim, key_size, bias=bias)
        self.W_v = nn.Linear(kv_in_dim, value_size, bias=bias)
        self.W_o = nn.Linear(value_size, q_in_dim, bias=bias)

    def forward(self, queries, keys, values):
        Q_p = transpose_qkv(self.W_q(queries), self.num_heads)
        K_p = transpose_qkv(self.W_k(keys), self.num_heads)
        V_p = transpose_qkv(self.W_v(values), self.num_heads)
        output = self.attention(Q_p, K_p, V_p)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class EncoderBlock(nn.Module):
    """Transformer encoder block."""
    def __init__(
        self,
        q_in_dim,
        kv_in_dim,
        key_size,
        query_size,
        value_size,
        num_heads,
        dropout,
        bias=False,
        **kwargs
    ):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(
            q_in_dim,
            kv_in_dim,
            key_size,
            query_size,
            value_size,
            num_heads,
            dropout,
            bias,
        )
        self.addnorm1 = AddNorm([q_in_dim], dropout)

    def forward(self, queries, keys, values):
        Y = self.addnorm1(queries, self.attention(queries, keys, values))
        return Y


class Encoder(nn.Module):
    """Base encoder class."""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, *args):
        raise NotImplementedError


class TransformerEncoder(Encoder):
    """Transformer encoder for feature encoding."""
    def __init__(
        self,
        q_in_dim,
        kv_in_dim,
        key_size,
        query_size,
        value_size,
        ffn_num_hiddens,
        num_heads,
        num_layers,
        dropout,
        bias=False,
        **kwargs
    ):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                "block" + str(i),
                EncoderBlock(
                    q_in_dim,
                    kv_in_dim,
                    key_size,
                    query_size,
                    value_size,
                    num_heads,
                    dropout,
                    bias,
                ),
            )

    def forward(self, p_feat, d_feat, *args):
        Y = p_feat
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            Y = blk(Y, d_feat, d_feat)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return Y


class GraphSAGE(nn.Module):
    """
    GraphSAGE: Graph neural network for node feature aggregation.
    
    A simplified and efficient implementation based on GCN principles:
    1. Uses the same adjacency matrix processing as GCN (no re-normalization)
    2. Simplified structure: single linear layer, removes LayerNorm and complex MLP
    3. Retains GraphSAGE core idea: concatenate self and aggregated features
    4. Uses residual connection for training stability
    """
    def __init__(self, p_feat_dim, d_feat_dim, n_hidden, dropout, num_layers=1):
        super(GraphSAGE, self).__init__()
        self.linear_p = nn.Linear(p_feat_dim, n_hidden)
        self.linear_d = nn.Linear(d_feat_dim, n_hidden)
        
        # Simplified: single linear layer (closer to GCN, but retains GraphSAGE concat idea)
        # Input: 2*n_hidden (self + aggregated), Output: n_hidden
        self.sage_linear = nn.Linear(2 * n_hidden, n_hidden)
        self.dropout = dropout
        
    def _mean_aggregate(self, x, adj):
        """
        Mean aggregation: uses the same adj processing as GCN.
        Directly uses adj matrix (adj is already normalized in main.ipynb).
        """
        # Directly use adj for aggregation (consistent with GCN, no re-normalization)
        aggregated = torch.mm(adj, x)
        return aggregated
    
    def forward(self, p_feat, d_feat, adj):
        num_p = p_feat.shape[0]
        p_feat = self.linear_p(p_feat)
        d_feat = self.linear_d(d_feat)
        x = torch.vstack((p_feat, d_feat))
        x_agg = self._mean_aggregate(x, adj)
        x_concat = torch.cat([x, x_agg], dim=1)
        x_new = self.sage_linear(x_concat)
        x = x + x_new 
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        p_feat = x[:num_p, :]
        d_feat = x[num_p:, :]
        return p_feat, d_feat


class Predictor(nn.Module):
    """Prediction head for association prediction."""
    def __init__(self):
        super(Predictor, self).__init__()

    def forward(self, p_feat, d_feat):
        res = p_feat.mm(d_feat.t())
        return F.sigmoid(res)


class PUTransGCN(nn.Module):
    """
    PUTransGCN: Main model for circRNA-disease association prediction.
    
    Architecture:
    1. DeepLncLoc: Extracts sequence features from circRNA
    2. GraphSAGE: Aggregates graph structure information
    3. TransformerEncoder: Encodes features with attention mechanism
    4. Predictor: Predicts association scores
    """
    def __init__(self, deep_lnc_loc, gcn, p_encoder, d_encoder, predictor, **kwargs):
        super(PUTransGCN, self).__init__(**kwargs)
        self.deep_lnc_loc = deep_lnc_loc
        self.gcn = gcn
        self.p_encoder = p_encoder
        self.d_encoder = d_encoder
        self.predictor = predictor

    def forward(self, p_kmers_id, d_feat, adj_mat):
        p_feat_dll = self.deep_lnc_loc(p_kmers_id)
        p_feat_gcn, d_feat_gcn = self.gcn(p_feat_dll, d_feat, adj_mat)
        p_enc_outputs = self.p_encoder(p_feat_gcn, d_feat_gcn)
        d_enc_outputs = self.d_encoder(d_feat_gcn, p_feat_gcn)
        return self.predictor(p_enc_outputs, d_enc_outputs)
