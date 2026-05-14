import numpy as np
import torch
import torch.nn.functional as F
import math
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter, softmax


class HGNNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(HGNNConv, self).__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, hyperedge_index):
        # x shape: [num_nodes, in_channels]
        # hyperedge_index shape: [2, num_incidences]

        # 防御性检查
        if hyperedge_index is None or hyperedge_index.size(1) == 0:
            # 返回一个经过线性变换的原始特征，以保证输出维度匹配
            return self.lin(x)

        num_nodes = x.size(0)
        num_hyperedges = hyperedge_index[1].max().item() + 1

        # 节点->超边, hyperedge_index[0]是节点索引, hyperedge_index[1]是超边索引
        node_idx, hyperedge_idx = hyperedge_index
        message_to_hyperedge = x[node_idx]
        hyperedge_emb = scatter(message_to_hyperedge, hyperedge_idx, dim=0, dim_size=num_hyperedges, reduce='mean')

        # 超边->节点
        message_to_node = hyperedge_emb[hyperedge_idx]
        node_emb_aggregated = scatter(message_to_node, node_idx, dim=0, dim_size=num_nodes, reduce='mean')

        out = self.lin(node_emb_aggregated)
        return out


class HypergraphGNN(torch.nn.Module):
    def __init__(self, d_model, n_layers=3):
        super().__init__()
        self.conv_list = nn.ModuleList(
            [HGNNConv(d_model, d_model) for _ in range(n_layers)]
        )
        # LayerNorm 稳定训练
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

    def forward(self, x, hyperedge_index):
        # 确保hyperedge_index非空
        if hyperedge_index is None or hyperedge_index.size(1) == 0:
            return x  # 如果没有超边，直接返回原始特征

        for conv, norm in zip(self.conv_list, self.norms):
            residual = x
            x = conv(x, hyperedge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = norm(x + residual)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, q, k, v, attn_mask=None, other=None):
        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(q.size(3))
        if other is not None:
            scores = scores + other
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, v)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads=1, dropout=0.5, d_v=64, d_k=64):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_v = d_v
        self.d_k = d_k
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_q, input_k, input_v, attn_mask=None, other=None):
        residual, batch_size = input_q, input_q.size(0)
        q = self.W_Q(input_q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_K(input_k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_V(input_v).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        context, attn = ScaledDotProductAttention()(q, k, v, attn_mask, other)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(context)
        output = self.dropout(output)
        return output, attn


class EmbeddingLayer(nn.Module):

    def __init__(self, config):
        super(EmbeddingLayer, self).__init__()
        self.config = config

        # embedding layer
        self.userEmbLayer = nn.Embedding(config.max_user_num, config.hidden_size, 0)
        self.locEmbLayer = nn.Embedding(config.max_loc_num, config.hidden_size, 0)
        self.geoEmbLayer = nn.Embedding(config.max_geo_num, config.hidden_size, 0)
        self.catEmbLayer = nn.Embedding(config.max_cat_num, config.hidden_size, 0)

        nn.init.normal_(self.userEmbLayer.weight, std=0.1)
        nn.init.normal_(self.locEmbLayer.weight, std=0.1)
        nn.init.normal_(self.geoEmbLayer.weight, std=0.1)
        nn.init.normal_(self.catEmbLayer.weight, std=0.1)

        self.gate_network = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Sigmoid()
        )

    def forward(self, user, traj, geo, time, traj_cat, long_traj, long_traj_cat, traj_graph, geo_graph):
        user_emb = self.userEmbLayer(user)
        traj_emb = self.locEmbLayer(traj)
        geo_emb = self.geoEmbLayer(geo)
        cat_emb = self.catEmbLayer(traj_cat)
        long_traj_cat_emb = self.catEmbLayer(long_traj_cat)
        long_traj_emb = self.locEmbLayer(long_traj)

        traj_graph.node_ids = traj_graph.x.clone()  # 保存原始ID
        traj_graph.x = self.locEmbLayer(traj_graph.x)  # 替换为嵌入

        geo_graph.node_ids = geo_graph.x.clone()
        geo_graph.x = self.geoEmbLayer(geo_graph.x)

        return user_emb, traj_emb, geo_emb, cat_emb, long_traj_emb, long_traj_cat_emb, traj_graph, geo_graph


class SpatialRoutineEncoder(nn.Module):

    def __init__(self, d_model, n_heads=4, dropout=0.5):
        super(SpatialRoutineEncoder, self).__init__()
        self.traj_conv = HypergraphGNN(d_model, 3)  # 位置图卷积
        self.geo_conv = HypergraphGNN(d_model, 3)  # 区域图卷积
        self.dropout1 = nn.Dropout()

    def forward(self, center_traj, traj_graph, geo_graph):
        traj_conv_out = self.traj_conv(traj_graph.x, traj_graph.hyperedge_index)
        geo_conv_out = self.geo_conv(geo_graph.x, geo_graph.hyperedge_index)

        # 更新图对象中的节点特征
        traj_graph.x = traj_conv_out
        geo_graph.x = geo_conv_out

        traj_global = global_mean_pool(traj_conv_out, traj_graph.batch)  # [batch, d]
        geo_global = global_mean_pool(geo_conv_out, geo_graph.batch)

        # 获取中心感知轨迹嵌入
        if hasattr(traj_graph, 'ptr') and traj_graph.ptr is not None:
            center_traj = center_traj + traj_graph.ptr[:-1].unsqueeze(1)

        # 从卷积后的节点特征中，根据中心点索引，提取出中心点序列的嵌入
        center_traj_emb = traj_conv_out[center_traj]
        center_traj_emb = self.dropout1(center_traj_emb)

        return center_traj_emb, traj_global, geo_global


class LocalWindowTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, window_size, dropout=0.2):
        super(LocalWindowTransformerLayer, self).__init__()
        self.window_size = window_size

        # 多头注意力
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=dropout)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

        # 归一化与 Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        seq_len = x.size(1)
        # 上三角全是 True，主对角线及以下全是 False
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        # 下三角中，偏离主对角线距离超过 window_size 的地方，也要变成 True (不可见)
        if self.window_size > 0:
            too_old_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device), diagonal=-self.window_size).bool()

            # 最终的 Mask：未来的不能看(mask) + 太老的不能看(too_old_mask)
            mask = mask | too_old_mask

        residual = x
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(residual + self.dropout1(attn_out))
        residual = x
        ffn_out = self.ffn(x)
        x = self.norm2(residual + self.dropout2(ffn_out))

        return x


class LocalIntentEncoder(nn.Module):
    def __init__(self, d_model, n_layers=1, n_heads=4, window_size=5, dropout=0.5):
        super(LocalIntentEncoder, self).__init__()
        self.d_model = d_model

        transformer_dim = d_model * 4

        self.layers = nn.ModuleList([
            LocalWindowTransformerLayer(
                d_model=transformer_dim,
                n_heads=n_heads,
                window_size=window_size,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        self.base_proj = nn.Linear(d_model, d_model * 2)
        self.center_fusion = nn.Linear(d_model * 5, d_model * 4)

        # 注意力融合模块
        self.attn = MultiHeadAttention(
            d_model=d_model * 4,
            n_heads=n_heads,
            dropout=dropout
        )

        self.w = nn.Parameter(torch.ones(2))
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 8, d_model * 4),  # 拼接后维度为 dim*2，输出维度为 dim
            nn.Sigmoid()
        )
        self.center_norm = nn.LayerNorm(d_model * 4)
        self.center_scale = nn.Parameter(torch.ones(1))

    def forward(self, user_emb, traj_emb, geo_emb, cat_emb, center_traj_emb, long_traj_emb, long_traj_cat_emb,
                dt, time, traj_global, geo_global):

        input = torch.concat([traj_emb, geo_emb, cat_emb, user_emb], dim=-1)
        transformer_output = input
        for layer in self.layers:
            transformer_output = layer(transformer_output)

        # center_input处理
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        center_input = F.relu(w1 * long_traj_emb + w2 * center_traj_emb)
        projected_base = self.base_proj(center_input)

        repeat_ratio = long_traj_emb.size(1) // traj_emb.size(1)
        if repeat_ratio > 0:
            user_emb_repeated = user_emb.repeat(1, repeat_ratio, 1)
        else:
            user_emb_repeated = user_emb[:, :projected_base.size(1), :]

        # 将 traj_global 和 geo_global 扩展到与 long_len 相同的长度
        traj_global_exp = traj_global.unsqueeze(1).expand(-1, projected_base.size(1), -1)  # [batch, long_len, d]
        geo_global_exp = geo_global.unsqueeze(1).expand(-1, projected_base.size(1), -1)

        center_input = torch.cat([
            center_input,
            long_traj_cat_emb,
            user_emb_repeated,
            traj_global_exp,
            geo_global_exp
        ], dim=-1)

        center_input = self.center_fusion(center_input)

        dt_expanded = dt.unsqueeze(1)

        # 注意力融合
        center_out, _ = self.attn(
            transformer_output,
            center_input,
            center_input,
            other=(1 / (1 + dt_expanded))
        )

        # gate_input = torch.cat([transformer_output, center_out], dim=-1)  # [B, L, 2*dim]
        # gate = self.gate_net(gate_input)  # [B, L, dim]
        # out = gate * transformer_output + (1 - gate) * center_out

        # center_out = self.center_scale * center_out

        # out = center_out
        out = transformer_output * torch.exp(-center_out)

        # out = transformer_output

        out = self.dropout2(out)

        return out


class TimeCategoryEncoder(nn.Module):
    def __init__(self, d_model, nhead=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.hour_emb = nn.Embedding(24, d_model)
        nn.init.normal_(self.hour_emb.weight, std=0.1)

        # 输入特征维度为 2*d（类别 + 小时）
        self.input_proj = nn.Linear(d_model * 2, d_model)  # 可选投影
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, cat_ids, hour_ids, cat_emb_layer):
        # 生成 padding mask（True 表示该位置是填充，需要被忽略）
        pad_mask = (cat_ids == 0)  # [batch, seq_len]

        cat_emb = cat_emb_layer(cat_ids)  # [batch, seq_len, d]
        hour_emb = self.hour_emb(hour_ids)

        x = torch.cat([cat_emb, hour_emb], dim=-1)  # [batch, seq_len, 2*d]
        x = self.input_proj(x)  # 投影回 d

        x = self.transformer(x, src_key_padding_mask=pad_mask)  # [batch, seq_len, d]

        # 池化时也需考虑mask
        x_masked = x * (~pad_mask).unsqueeze(-1).float()  # 将填充位置置零
        seq_len_effective = (~pad_mask).sum(dim=1, keepdim=True).float()  # [batch, 1]
        x = x_masked.sum(dim=1) / (seq_len_effective + 1e-8)  # [batch, d]
        return x


class PoiModelnew(nn.Module):

    def __init__(self, config):
        super(PoiModelnew, self).__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.max_seq_len = config.max_sequence_length

        self.EmbeddingLayer = EmbeddingLayer(config)
        self.SpatialRoutineEncoder = SpatialRoutineEncoder(config.hidden_size)

        self.LocalIntentEncoder = LocalIntentEncoder(
            d_model=self.hidden_size,
            n_layers=3,
            n_heads=4,
            window_size=5,
            dropout=0.5
        )

        self.time_proj = nn.Linear(4, self.hidden_size)
        self.time_gate = nn.Sequential(
            nn.Linear(self.hidden_size * 5, self.hidden_size * 4),
            nn.Sigmoid()
        )

        self.fc_traj = nn.Linear(config.hidden_size * 5, config.max_loc_num)  # POI预测
        self.fc_geo = nn.Linear(config.hidden_size * 5, config.max_geo_num)  # 区域预测
        self.fc_cat = nn.Linear(config.hidden_size * 5, config.max_cat_num)  # 类别预测

        self.hourEmb = nn.Embedding(24, config.hidden_size)
        nn.init.normal_(self.hourEmb.weight, std=0.1)

        self.time_cat_encoder = TimeCategoryEncoder(
            d_model=config.hidden_size,
            nhead=4,
            num_layers=2,
            dropout=0.3
        )

    def forward(self, user, traj, geo, time, traj_cat, center_traj, long_traj, long_traj_cat, dt, traj_graph,
                geo_graph, time_feat, hour):
        # user/traj/geo shape: (batch_size, max_sequence_length)
        # center_traj/long_traj shape: (batch_size, long_sequence_length)
        # dt shape: (batch_size, max_sequence_length, max_sequence_length)
        # user_emb/traj_emb/geo_emb shape: (batch_size, max_sequence_length, hidden_size)

        user_emb, traj_emb, geo_emb, cat_emb, long_traj_emb, long_traj_cat_emb, traj_graph, geo_graph = self.EmbeddingLayer(
            user, traj, geo, time, traj_cat, long_traj, long_traj_cat, traj_graph, geo_graph)

        # center_traj_emb shape: (batch_size, long_sequence_length, hidden_size)
        center_traj_emb, traj_global, geo_global = self.SpatialRoutineEncoder(
            center_traj, traj_graph, geo_graph)

        short_enc_out = self.LocalIntentEncoder(
            user_emb, traj_emb, geo_emb, cat_emb, center_traj_emb, long_traj_emb, long_traj_cat_emb,
            dt, time, traj_global, geo_global
        )

        # 处理时间特征time_feat shape: [Batch, Seq_len, 4] -> [Batch, Seq_len, hidden_size]
        time_context = self.time_proj(time_feat)

        # 用当前的time_context去审查轨迹特征
        gate_input = torch.cat([short_enc_out, time_context], dim=-1)  # shape: [B, S, hidden_size * 2]

        # 计算门控权重
        time_weights = self.time_gate(gate_input)  # shape: [B, S, hidden_size]
        time_aware_rep = short_enc_out * time_weights  # 特征过滤

        time_cat_emb = self.time_cat_encoder(
            cat_ids=traj_cat,
            hour_ids=hour,
            cat_emb_layer=self.EmbeddingLayer.catEmbLayer
        )  # [batch]

        # 将time_cat_emb扩展到与time_aware_rep相同的序列长度
        time_cat_expanded = time_cat_emb.unsqueeze(1).expand(-1, time_aware_rep.size(1), -1)

        # 拼接得到三维表示
        final_rep = torch.cat([time_aware_rep, time_cat_expanded], dim=-1)  # [batch, seq_len, 4*d+d]

        pred_traj = self.fc_traj(final_rep)
        pred_geo = self.fc_geo(final_rep)
        pred_cat = self.fc_cat(final_rep)

        return pred_traj, pred_geo, pred_cat
