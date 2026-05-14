import os
import torch
import numpy as np
import pandas as pd
import pygeohash as gh
import logging
import random


def geohash_encode(checkins, num=5):  # 新增geohash列
    checkins['geohash'] = checkins.apply(
        lambda x: gh.encode(x['latitude'], x['longitude'], precision=num), axis=1)
    return checkins


def geohash_neighbors(geohash):  # 计算一个geohash区域的及其8个邻居区域
    neighbors = []
    lat_range, lon_range = 180, 360
    x, y = gh.decode(geohash)  # 解码geohash获取中心点坐标，x=纬度, y=经度
    num = len(geohash) * 5
    dx = lat_range / (2**(num // 2))
    dy = lon_range / (2**(num - num // 2))
    for i in range(1, -2, -1):
        for j in range(-1, 2):
            # 计算相邻网格的中心点坐标，重新编码为geohash
            neighbors.append(gh.encode(x + i * dx, y + j * dy, num // 5))
    return neighbors


def init_seed(seed=256):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    consol_handler = logging.StreamHandler()
    consol_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '[%(asctime)s %(filename)s line:%(lineno)d process:%(process)d] %(levelname)s: %(message)s'
    )
    consol_handler.setFormatter(formatter)
    logger.addHandler(consol_handler)
    return logger


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    difflat = lat2 - lat1
    difflon = lon2 - lon1

    a = np.sin(difflat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(difflon / 2)**2
    distance = 2 * np.arcsin(np.sqrt(a)) * R
    return distance


def split_user_train_test(checkins, train_size):
    def split_train_test(df):
        n_train = int(len(df) * train_size)
        checkins_train.append(df.iloc[:n_train])
        checkins_test.append(df.iloc[n_train - 1:])

    checkins_train, checkins_test = [], []
    _ = checkins.groupby('user_id').apply(split_train_test)
    checkins_train = pd.concat(checkins_train).reset_index(drop=True)
    checkins_test = pd.concat(checkins_test).reset_index(drop=True)
    return checkins_train, checkins_test


def calculate_acc(pred, labels):
    pred = pred.view(-1, pred.shape[2])
    labels = labels.view(-1).unsqueeze(0)
    result = torch.zeros(5, labels.shape[1], device=pred.device)
    pred_val, pred_poi = pred.topk(20, dim=1, sorted=True)
    recall = torch.stack([labels == pred_poi[:, i] for i in range(20)])
    result[0] = recall[:1].sum(dim=0)
    result[1] = recall[:5].sum(dim=0)
    result[2] = recall[:10].sum(dim=0)
    result[3] = recall[:20].sum(dim=0)

    score = pred.gather(dim=1, index=labels.T)
    result[4] = 1 / (1 + (pred > score).sum(dim=1))
    return result


def process_timestamp(timestamps):
    dt = torch.tensor([pd.to_datetime(ts, unit='s') for ts in timestamps.cpu().numpy()])
    hour = dt[:, :, 3].unsqueeze(-1) / 23.0
    weekday = dt[:, :, 4].unsqueeze(-1) / 6.0
    is_workday = (weekday < 5).float().unsqueeze(-1)

    time_feat = torch.cat([hour, weekday, is_workday], dim=-1)
    return time_feat.to(timestamps.device)


def drop_checkins_global(train_df, drop_ratio, seed=42):  # 用于鲁棒性实验
    np.random.seed(seed)
    n_total = len(train_df)
    n_keep = int(n_total * (1 - drop_ratio))

    # 随机选择保留的索引
    keep_idx = np.random.choice(train_df.index, size=n_keep, replace=False)
    result = train_df.loc[keep_idx].reset_index(drop=True)

    print(f"[Robustness] 原始训练集: {n_total} 条签到")
    print(f"[Robustness] 删除后训练集: {len(result)} 条签到")
    print(f"[Robustness] 实际删除比例: {1 - len(result) / n_total: .2%}")
    print(f"[Robustness] 剩余用户数: {result['user_id'].nunique()} / {train_df['user_id'].nunique()}")
    print(f"[Robustness] 剩余POI数: {result['location_id'].nunique()} / {train_df['location_id'].nunique()}")
    return result
