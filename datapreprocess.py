import numpy as np
import torch
import torch.nn.functional as F
import tools
import pandas as pd
from torch import LongTensor as LT
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import index_to_mask
from collections import Counter
from collections import defaultdict


class HyperData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'hyperedge_index':
            # self.num_nodes 是 PyG 的标准属性，可以安全使用。它代表当前这个子图的节点数。
            # 这里的value是hyperedge_index本身,从它动态计算出超边的数量。
            if value.numel() == 0:
                num_h_edges = 0
            else:
                # 从hyperedge_index 的第二行（超边索引）找到最大值来确定超边数量。
                num_h_edges = value[1].max().item() + 1
            # 返回一个张量，告诉dataLoader节点索引需要增加self.num_nodes,超边索引需要增加 num_h_edges
            return torch.tensor([[self.num_nodes], [num_h_edges]])
        return super().__inc__(key, value, *args, **kwargs)


class MASHDataset(Dataset):

    def __init__(self, config, checkins, mapping):
        super(MASHDataset, self).__init__()
        self.config = config
        self.checkins = checkins
        self.loc2lat, self.loc2lon, self.geo2neighbor_loc = mapping['loc2lat'], mapping['loc2lon'], mapping[
            'geo2neighbor_loc']
        self.loc2cat = mapping['loc2cat']
        self.checkins_split()

    def checkins_split(self):
        def collect_fn(df):
            max_sequence_length = self.config.max_sequence_length  # 序列分割

            # Create batch pt(批次指针)
            cnt, remain = divmod(len(df) - 1, max_sequence_length)
            user_start_ptr.extend([len(user)] * cnt)
            batch_length.extend([max_sequence_length] * cnt)
            batch_ptr.extend([len(user) + i * max_sequence_length for i in range(cnt)])

            user.extend(df['user_id'].values)  # 用户id
            traj.extend(df['location_id'].values)  # 位置id
            time.extend(df['time_unix'].values)  # 时间戳
            geo.extend(df['geohash_id'].values)  # geohashID
            traj_cat.extend(df['category_id'].values)  # POI类别

            cur_user = df.iloc[0]['user_id']
            # 创建双超图
            traj_graph[cur_user] = self.create_hypergraph(df['location_id'].values, 'loc',
                                                          self.config.alpha)
            geo_graph[cur_user] = self.create_hypergraph(df['geohash_id'].values, 'geo',
                                                         self.config.beta)

            center_traj.extend(
                self.local_center_seq(traj_graph[cur_user], df['location_id'].values))

        user, traj, time, geo, traj_cat = [], [], [], [], []
        user_start_ptr, batch_ptr, batch_length = [], [], []
        center_traj = []
        traj_graph, geo_graph = {}, {}
        # 按用户分组处理
        _ = self.checkins.groupby('user_id').apply(collect_fn)

        self.user, self.traj, self.time, self.geo, self.traj_cat = LT(user), LT(traj), LT(time), LT(geo), LT(traj_cat)
        self.user_start_ptr, self.batch_ptr, self.batch_length = user_start_ptr, batch_ptr, batch_length
        self.center_traj = LT(center_traj)
        self.traj_graph, self.geo_graph = traj_graph, geo_graph

        self.negihbors_ptr = []
        # 获取邻域POI
        for idx in range(len(batch_ptr)):
            start, end = batch_ptr[idx], batch_ptr[idx] + batch_length[idx]
            long_start = max(user_start_ptr[idx], end - self.config.long_sequence_length)
            long_geo = self.geo[long_start:end]
            negihbors = self.get_neighborhood_loc(long_geo, self.config.min_neighborhood_num)
            self.negihbors_ptr.append(negihbors)

    def get_neighborhood_loc(self, geo_seq, num=1000):
        geo2neighbor_loc = self.geo2neighbor_loc
        ne_loc = torch.concat([geo2neighbor_loc[geo] for geo in geo_seq.numpy()]).unique()

        # 检查数量，如果不够则进行随机补充
        if len(ne_loc) < num:
            k = num - len(ne_loc)
            existing_samples = set(ne_loc.numpy())
            population = [i for i in range(1, self.config.max_loc_num) if i not in existing_samples]

            if len(population) < k:
                k = len(population)

            if k > 0:
                ne_random = np.random.choice(population, size=k, replace=False)
                ne_loc = torch.cat([ne_loc, torch.from_numpy(ne_random)]).unique()

        return ne_loc

    def create_hypergraph(self, seq, g_type, rate=0.05, co_occurrence_window=20):
        seq = LT(seq)
        # 节点定义和频率计算
        graph = HyperData(x=seq.unique()) if g_type == 'loc' else Data(x=seq.unique())
        node2idx = {node: i for i, node in enumerate(graph.x.numpy())}

        graph.freq = torch.zeros_like(graph.x)
        counts = Counter(seq.numpy())
        for node, cnt in counts.items():
            if node in node2idx:
                graph.freq[node2idx[node]] = cnt

        # 识别核心节点
        while (graph.freq >= len(seq) * rate).sum() == 0 and rate > 1e-6:
            rate /= 2
        graph.thr = torch.zeros_like(graph.x) + len(seq) * rate
        c_idxs = (graph.freq >= graph.thr).nonzero().view(-1)

        num_nodes = len(graph.x)

        if len(c_idxs) == 0:
            # 如果没有中心点，构造一个空的、但结构完整的图
            if g_type == 'loc':
                graph.hyperedge_index = torch.empty((2, 0), dtype=torch.long)
                graph.num_nodes = num_nodes
                graph.num_hyperedges = 0
                graph.center = torch.zeros(num_nodes, dtype=torch.long)
            else:  # geo图保持原样
                graph.edge_index = torch.empty((2, 0), dtype=torch.long)
            return graph

        center_nodes_ingraph_idx = c_idxs
        num_hyperedges = len(center_nodes_ingraph_idx)

        # 构建双超图超边关联
        hyperedge_assignments = torch.zeros(num_nodes, dtype=torch.long)

        if g_type == 'loc':  # loc分支：使用物理距离
            center_nodes = graph.x[center_nodes_ingraph_idx]
            c_lons, c_lats = self.loc2lon[center_nodes], self.loc2lat[center_nodes]
            for i, node in enumerate(graph.x):
                lon, lat = self.loc2lon[node], self.loc2lat[node]
                dis = tools.haversine_distance(lat, lon, c_lats, c_lons)
                hyperedge_assignments[i] = dis.argmin()

            # 为loc图构造并附加属性
            graph.center = center_nodes_ingraph_idx[hyperedge_assignments]
            node_indices = torch.arange(num_nodes)
            graph.hyperedge_index = torch.stack([node_indices, hyperedge_assignments], dim=0)
            graph.num_nodes = num_nodes
            graph.num_hyperedges = num_hyperedges

        elif g_type == 'geo':  # geo分支：使用共现频率
            # 创建一个从图内索引到核心区域索引的映射，便于快速查找
            core_geo_indices = set(center_nodes_ingraph_idx.numpy())
            core_geo_map = {core_idx: i for i, core_idx in enumerate(center_nodes_ingraph_idx.numpy())}

            # 初始化共现分数矩阵
            co_occurrence_scores = torch.zeros((num_nodes, num_hyperedges))

            # 使用滑动窗口计算共现
            seq_ingraph_idx = np.vectorize(node2idx.get)(seq.numpy())
            for i in range(len(seq_ingraph_idx) - co_occurrence_window + 1):
                window = seq_ingraph_idx[i: i + co_occurrence_window]

                # 找出窗口内的核心geo和普通geo
                window_core_geos = {idx for idx in window if idx in core_geo_indices}
                window_normal_geos = {idx for idx in window if idx not in core_geo_indices}

                # 为每个 "普通-核心" 对增加分数
                for normal_idx in window_normal_geos:
                    for core_idx in window_core_geos:
                        # 找到核心geo对应的超边索引
                        hyperedge_idx = core_geo_map[core_idx]
                        co_occurrence_scores[normal_idx, hyperedge_idx] += 1

            # 为每个节点分配超边,对于普通节点，分配给共现分数最高的超边,对于核心节点，分配给自己代表的超边
            for i in range(num_nodes):
                if i in core_geo_indices:
                    hyperedge_assignments[i] = core_geo_map[i]
                else:
                    # 如果一个普通节点从未与任何核心节点共现，随机分配或分配给0
                    if co_occurrence_scores[i].sum() == 0:
                        hyperedge_assignments[i] = 0
                    else:
                        hyperedge_assignments[i] = co_occurrence_scores[i].argmax()

            # 为geo图构造并附加属性
            graph.center = center_nodes_ingraph_idx[hyperedge_assignments]
            node_indices = torch.arange(num_nodes)
            graph.hyperedge_index = torch.stack([node_indices, hyperedge_assignments], dim=0)
            graph.num_nodes = num_nodes
            graph.num_hyperedges = num_hyperedges

            if hasattr(graph, 'edge_index'):
                del graph.edge_index
        return graph

    def local_center_seq(self, graph, seq):
        center_mapping = dict(zip(graph.x.numpy(), graph.center.numpy()))
        center_seq = np.vectorize(center_mapping.get)(seq)
        return center_seq

    def extract_time_features(self, time_unix):
        dt = pd.to_datetime(time_unix, unit='s')
        hour = dt.hour.values
        weekday = dt.weekday.values  # Monday=0, Sunday=6

        # 小时特征 (周期为24)
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)

        # 星期特征 (周期为7)
        weekday_sin = np.sin(2 * np.pi * weekday / 7.0)
        weekday_cos = np.cos(2 * np.pi * weekday / 7.0)

        # 拼接为4维特征
        time_feat = np.concatenate([
            hour_sin.reshape(-1, 1),
            hour_cos.reshape(-1, 1),
            weekday_sin.reshape(-1, 1),
            weekday_cos.reshape(-1, 1)
        ], axis=1)

        return torch.FloatTensor(time_feat)

    def __len__(self):
        return len(self.batch_ptr)

    def __getitem__(self, idx):
        # 定义序列范围
        short_start = self.batch_ptr[idx]
        short_end = short_start + self.batch_length[idx]
        long_start = max(self.user_start_ptr[idx], short_end - self.config.long_sequence_length)

        # 短序列
        user = self.user[short_start:short_end]
        traj = self.traj[short_start:short_end]
        geo = self.geo[short_start:short_end]
        time = self.time[short_start:short_end]
        label_traj = self.traj[short_start + 1:short_end + 1]
        label_geo = self.geo[short_start + 1:short_end + 1]
        traj_cat = self.traj_cat[short_start:short_end]
        label_cat = self.traj_cat[short_start + 1:short_end + 1]

        # 长序列
        long_traj = self.traj[long_start:short_start]
        center_traj = self.center_traj[long_start:short_start]
        long_time = self.time[long_start:short_start]
        long_traj_cat = self.traj_cat[long_start:short_start]
        time_feat = self.extract_time_features(time.numpy())

        # 健壮性处理
        if len(long_time) > 0 and len(time) > 0:
            dt = torch.stack([abs(long_time - t) for t in time])
        else:
            # 创建一个正确形状的空矩阵
            dt = torch.zeros(len(time), len(long_time), dtype=torch.long)

        # Padding短序列
        if len(user) != self.config.max_sequence_length:
            pad_len = self.config.max_sequence_length - len(user)
            user = F.pad(user, (0, pad_len))
            traj = F.pad(traj, (0, pad_len))
            geo = F.pad(geo, (0, pad_len))
            time = F.pad(time, (0, pad_len))
            label_traj = F.pad(label_traj, (0, pad_len))
            label_geo = F.pad(label_geo, (0, pad_len))
            traj_cat = F.pad(traj_cat, (0, pad_len))
            label_cat = F.pad(label_cat, (0, pad_len))
            time_feat = F.pad(time_feat, (0, 0, 0, pad_len))
            # padding dt 矩阵的短序列维度
            dt = F.pad(dt, (0, 0, 0, pad_len))  # (pad_left, pad_right, pad_top, pad_bottom)

        # Padding长序列
        if len(long_traj) != self.config.long_sequence_length:
            pad_len = self.config.long_sequence_length - len(long_traj)
            long_traj = F.pad(long_traj, (0, pad_len))
            center_traj = F.pad(center_traj, (0, pad_len))
            long_traj_cat = F.pad(long_traj_cat, (0, pad_len))
            # padding dt 矩阵的长序列维度
            dt = F.pad(dt, (0, pad_len, 0, 0))

        negihbors_mask = ~index_to_mask(self.negihbors_ptr[idx], size=self.config.max_loc_num)
        negihbors_mask[0] = True

        # 双超图
        traj_graph = self.traj_graph[int(user[0])]
        geo_graph = self.geo_graph[int(user[0])]

        hour = (torch.div(time, 3600, rounding_mode='floor')) % 24

        return user, traj, geo, time, traj_cat, center_traj, long_traj, long_traj_cat, dt, label_traj, label_geo, \
            label_cat, negihbors_mask, traj_graph, geo_graph, time_feat, hour

