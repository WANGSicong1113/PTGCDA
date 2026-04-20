"""
模型定义文件 - 包含Attention和无Attention版本

使用说明：
1. TransformerEncoder: 原始版本，使用MultiHeadAttention
2. IdentityEncoder: 无Attention版本，直接返回输入（完全不做变换）
3. LinearEncoder: 无Attention版本，使用简单线性层（带LayerNorm和Dropout）

消融实验使用方法：
在main.ipynb中，将：
    p_encoder = TransformerEncoder(...)
    d_encoder = TransformerEncoder(...)
    
改为：
    p_encoder = IdentityEncoder(...)  # 或 LinearEncoder(...)
    d_encoder = IdentityEncoder(...)  # 或 LinearEncoder(...)

注意：所有Encoder的接口完全兼容，可以直接替换使用。
"""

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


def adaptive_avg_pool1d_mps_compatible(x, output_size):
    """
    MPS 兼容的 AdaptiveAvgPool1d 实现
    在 MPS 设备上，如果遇到不支持的操作，临时移到 CPU 执行
    """
    if x.device.type == "mps":
        input_size = x.shape[2]
        if input_size % output_size != 0:
            # MPS 不支持 interpolate 的 linear 模式，需要移到 CPU
            # 将数据移到 CPU，执行操作，再移回 MPS
            x_cpu = x.cpu()
            result_cpu = F.interpolate(x_cpu, size=output_size, mode='linear', align_corners=False)
            return result_cpu.to(x.device)
        else:
            # 可以整除，尝试使用标准的 AdaptiveAvgPool1d
            try:
                return F.adaptive_avg_pool1d(x, output_size)
            except (NotImplementedError, RuntimeError):
                # 如果 MPS 不支持，也移到 CPU 执行
                x_cpu = x.cpu()
                result_cpu = F.adaptive_avg_pool1d(x_cpu, output_size)
                return result_cpu.to(x.device)
    else:
        # 非 MPS 设备，使用标准实现
        return F.adaptive_avg_pool1d(x, output_size)


class DeepLncLoc(nn.Module):
    def __init__(self, w2v_emb, dropout, merge_win_size, context_size_list, out_size):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(w2v_emb, freeze=False)
        self.dropout = nn.Dropout(dropout)
        self.merge_win_size = merge_win_size  # 保存输出大小用于 MPS 兼容性处理
        # 注意：不在 __init__ 中创建 AdaptiveAvgPool1d，而是在 forward 中动态处理
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
        x = x.transpose(1, 2)
        # 使用 MPS 兼容的池化函数
        x = adaptive_avg_pool1d_mps_compatible(x, self.merge_win_size)
        x = [conv(x).squeeze(dim=2) for conv in self.con_list]
        x = torch.cat(x, dim=1)
        return x


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        return F.elu(h_prime)

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[: self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features :, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class GAT(nn.Module):
    def __init__(
        self,
        p_feat_dim,
        d_feat_dim,
        hidden_dim,
        out_dim,
        dropout,
        alpha,
        nheads,
    ):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.bn_p = nn.BatchNorm1d(p_feat_dim)
        self.bn_d = nn.BatchNorm1d(d_feat_dim)
        self.linear_p = nn.Linear(p_feat_dim, hidden_dim)
        self.linear_d = nn.Linear(d_feat_dim, hidden_dim)
        assert out_dim % nheads == 0
        nhid_per_head = int(out_dim / nheads)
        self.layer1 = [
            GraphAttentionLayer(hidden_dim, nhid_per_head, dropout=dropout, alpha=alpha)
            for _ in range(nheads)
        ]
        for i, head in enumerate(self.layer1):
            self.add_module("layer1_head_{}".format(i), head)

        self.out_att = GraphAttentionLayer(
            nhid_per_head * nheads, out_dim, dropout=dropout, alpha=alpha
        )

    def forward(self, p_feat, d_feat, adj):
        p_feat_hidden = self.linear_p(self.bn_p(p_feat))
        d_feat_hidden = self.linear_d(self.bn_d(d_feat))
        x = torch.cat((p_feat_hidden, d_feat_hidden), dim=0)

        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.layer1], dim=1)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        p_feat = x[: p_feat.shape[0], :]
        d_feat = x[p_feat.shape[0] :, :]
        return p_feat, d_feat


class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


# @save
class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


def transpose_qkv(X, num_heads):
    X = X.reshape(X.shape[0], num_heads, -1)
    X = X.permute(1, 0, 2)
    return X


def transpose_output(X, num_heads):
    X = X.permute(1, 0, 2)
    return X.reshape(X.shape[0], -1)


def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    mask = (
        torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :]
        < valid_len[:, None]
    )
    X[~mask] = value
    return X


class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q_p, K_p, V_p):
        d = Q_p.shape[-1]
        scores = torch.bmm(Q_p, K_p.transpose(1, 2)) / np.sqrt(d)
        self.attention_weights = nn.functional.softmax(scores, dim=-1)
        return torch.bmm(self.dropout(self.attention_weights), V_p)


class MultiHeadAttention(nn.Module):
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
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, *args):
        raise NotImplementedError


# @save
class TransformerEncoder(Encoder):
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
        
        # ========== 消融实验支持：当 num_layers=0 或 num_heads=0 时，不使用attention ==========
        self.use_attention = (num_layers > 0) and (num_heads > 0)
        
        if not self.use_attention:
            # 无attention模式：使用简单的线性层（可选）或直接返回
            self.use_linear = True  # 设置为False则完全不做变换（Identity）
            if self.use_linear:
                # 使用简单的线性层 + LayerNorm + Dropout（类似LinearEncoder）
                self.linear = nn.Linear(q_in_dim, q_in_dim, bias=bias)
                self.layer_norm = nn.LayerNorm(q_in_dim)
                self.dropout_layer = nn.Dropout(dropout)
            # 不需要创建attention blocks
            self.blks = nn.Sequential()
            self.attention_weights = []
        else:
            # 正常attention模式：创建Transformer blocks
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
            self.attention_weights = [None] * num_layers

    def forward(self, p_feat, d_feat, *args):
        if not self.use_attention:
            # 无attention模式：直接返回或使用简单线性变换
            if self.use_linear:
                # 使用线性层 + LayerNorm + Dropout + 残差连接
                output = self.linear(p_feat)
                output = self.layer_norm(self.dropout_layer(output) + p_feat)
                return output
            else:
                # 完全不做变换，直接返回（Identity）
                return p_feat
        else:
            # 正常attention模式：使用Transformer blocks
            Y = p_feat
            for i, blk in enumerate(self.blks):
                Y = blk(Y, d_feat, d_feat)
                self.attention_weights[i] = blk.attention.attention.attention_weights
            return Y


# ========== 无Attention版本：IdentityEncoder ==========
class IdentityEncoder(Encoder):
    """
    无Attention版本的Encoder - 用于消融实验
    直接返回输入特征，不做任何变换
    接口与TransformerEncoder完全兼容
    """
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
        super(IdentityEncoder, self).__init__(**kwargs)
        # 为了保持接口兼容性，接受所有参数但不使用
        # 可以添加一个可选的线性层来保持维度一致
        self.use_linear = False  # 设置为True可以使用简单的线性变换
        if self.use_linear:
            self.linear = nn.Linear(q_in_dim, q_in_dim)
            self.dropout = nn.Dropout(dropout)
        else:
            # 完全不做变换，直接返回
            pass

    def forward(self, p_feat, d_feat, *args):
        """
        直接返回输入特征，忽略d_feat
        与TransformerEncoder的接口保持一致
        """
        if self.use_linear:
            # 可选：使用简单的线性层（带dropout）
            return self.dropout(self.linear(p_feat))
        else:
            # 完全不做变换，直接返回
            return p_feat


# ========== 无Attention版本：LinearEncoder ==========
class LinearEncoder(Encoder):
    """
    无Attention版本的Encoder - 使用简单线性层替代
    比IdentityEncoder稍微复杂一点，但仍然没有attention机制
    接口与TransformerEncoder完全兼容
    """
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
        super(LinearEncoder, self).__init__(**kwargs)
        # 使用简单的线性层 + LayerNorm + Dropout
        self.linear = nn.Linear(q_in_dim, q_in_dim, bias=bias)
        self.layer_norm = nn.LayerNorm(q_in_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, p_feat, d_feat, *args):
        """
        使用简单的线性变换，忽略d_feat和attention机制
        与TransformerEncoder的接口保持一致
        """
        # 线性变换
        output = self.linear(p_feat)
        # LayerNorm + Dropout + 残差连接
        output = self.layer_norm(self.dropout(output) + p_feat)
        return output


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class GCN(nn.Module):
    def __init__(self, p_feat_dim, d_feat_dim, n_hidden, dropout):
        super(GCN, self).__init__()
        self.linear_p = nn.Linear(p_feat_dim, n_hidden)
        self.linear_d = nn.Linear(d_feat_dim, n_hidden)

        self.gc1 = GraphConvolution(n_hidden, n_hidden)
        self.dropout = dropout

    def forward(self, p_feat, d_feat, adj):
        p_feat = self.linear_p(p_feat)
        d_feat = self.linear_d(d_feat)
        x = torch.vstack((p_feat, d_feat))
        x = torch.nn.functional.relu(self.gc1(x, adj))
        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        p_feat = x[: p_feat.shape[0], :]
        d_feat = x[p_feat.shape[0] :, :]
        return p_feat, d_feat


class GraphSAGE(nn.Module):
    """
    简化高效版GraphSAGE - 基于GCN的成功经验优化
    关键改进：
    1. 使用与GCN相同的adj处理方式（不重新归一化）
    2. 简化结构：单层线性层，移除LayerNorm和复杂MLP
    3. 保留GraphSAGE核心思想：concat自身和聚合特征
    4. 使用残差连接稳定训练
    接口与GCN完全兼容，可以直接替换。
    """
    def __init__(self, p_feat_dim, d_feat_dim, n_hidden, dropout, num_layers=1):
        super(GraphSAGE, self).__init__()
        self.linear_p = nn.Linear(p_feat_dim, n_hidden)
        self.linear_d = nn.Linear(d_feat_dim, n_hidden)
        
        # 简化：单层线性层（更接近GCN，但保留GraphSAGE的concat思想）
        # 输入：2*n_hidden (自身 + 聚合), 输出：n_hidden
        self.sage_linear = nn.Linear(2 * n_hidden, n_hidden)
        self.dropout = dropout
        
    def _mean_aggregate(self, x, adj):
        """
        Mean聚合: 使用与GCN相同的adj处理方式
        直接使用adj矩阵（adj在main.ipynb中已经归一化过）
        """
        # 直接使用adj进行聚合（与GCN一致，不重新归一化）
        aggregated = torch.mm(adj, x)
        return aggregated
    
    def forward(self, p_feat, d_feat, adj):
        # 保存原始circRNA数量用于后续分离
        num_p = p_feat.shape[0]
        
        # 初始特征变换
        p_feat = self.linear_p(p_feat)
        d_feat = self.linear_d(d_feat)
        x = torch.vstack((p_feat, d_feat))
        
        # GraphSAGE前向传播
        # 1. Mean聚合邻居特征（使用与GCN相同的adj）
        x_agg = self._mean_aggregate(x, adj)
        
        # 2. 拼接自身特征和聚合特征（GraphSAGE核心思想）
        x_concat = torch.cat([x, x_agg], dim=1)
        
        # 3. 通过线性层变换（简化，移除复杂的MLP和LayerNorm）
        x_new = self.sage_linear(x_concat)
        
        # 4. 残差连接（稳定训练，加速收敛）
        x = x + x_new  # 残差连接：自身特征 + 变换后的特征
        
        # 5. 激活函数
        x = torch.nn.functional.relu(x)
        
        # 6. Dropout
        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        
        # 分离回p_feat和d_feat
        p_feat = x[:num_p, :]
        d_feat = x[num_p:, :]
        return p_feat, d_feat


class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()

    def forward(self, p_feat, d_feat):
        res = p_feat.mm(d_feat.t())
        return F.sigmoid(res)


class PUTransGCN(nn.Module):
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