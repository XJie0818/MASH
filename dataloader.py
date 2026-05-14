import pandas as pd
import torch
import tools
from datapreprocess import MASHDataset
from torch_geometric.loader import DataLoader
from pathlib import Path


class MASHdataloader():
    def __init__(self, config):
        self.config = config
        self.database = Path(config.database)
        if not self.database.exists():
            self.database.mkdir()

    def load(self, dataset, file):
        self.checkins = self.load_checkins(dataset, file)

        self.dataset = dataset
        setattr(self.config, 'max_user_num', self.user_count() + 1)
        setattr(self.config, 'max_loc_num', self.location_count() + 1)
        setattr(self.config, 'max_geo_num', self.geohash_count() + 1)
        setattr(self.config, 'max_cat_num', self.category_count() + 1)
        return self.checkins

    def create_POI_mapping(self):
        def split_loc(df):
            loc2lat.append(df['latitude'].values[0])
            loc2lon.append(df['longitude'].values[0])

        loc2lat, loc2lon = [0], [0]
        _ = self.checkins.groupby('location_id').apply(split_loc)
        loc2lat, loc2lon = torch.Tensor(loc2lat), torch.Tensor(loc2lon)

        geo2id = self.checkins.set_index('geohash')['geohash_id'].to_dict()

        # 获取每个geohash区域包含的POI和邻居区域
        def geo2loc_collect(df):
            geohash, geohash_id = df.iloc[0][['geohash', 'geohash_id']]
            # 映射1：geohash_id -> 该区域内的所有位置ID
            geo2loc[geohash_id] = torch.from_numpy(df['location_id'].unique())
            # 映射2：geohash_id -> 相邻geohash区域的ID
            geo2neighbor[geohash_id] = [
                geo2id[ne] for ne in tools.geohash_neighbors(geohash)
                if ne in geo2id.keys()
            ]

        geo2loc, geo2neighbor = {}, {}
        _ = self.checkins.groupby('geohash').apply(geo2loc_collect)

        # 计算geohash到邻居位置的映射
        geo2neighbor_loc = {}
        for key, val in geo2neighbor.items():
            geo2neighbor_loc[key] = torch.concat([geo2loc[geo] for geo in val]).unique()

        # 创建loc_id -> cat_id的映射
        loc_cat_map_df = self.checkins[['location_id', 'category_id']].drop_duplicates()
        loc2cat_series = loc_cat_map_df.set_index('location_id')['category_id']
        max_loc_num = self.config.max_loc_num
        loc2cat_tensor = torch.zeros(max_loc_num, dtype=torch.long)
        loc2cat_tensor[loc2cat_series.index] = torch.from_numpy(loc2cat_series.values)

        # 将所有映射打包成一个字典返回
        mapping = {
            'loc2lat': loc2lat,
            'loc2lon': loc2lon,
            'geo2neighbor_loc': geo2neighbor_loc,
            'loc2cat': loc2cat_tensor
        }

        return mapping

    def create_dataset(self, mode, dataset, drop_ratio=0.0):
        print(f"create_dataset called with drop_ratio={drop_ratio}")

        # 缓存训练集/测试集的路径
        drop_suffix = f"_drop{int(drop_ratio * 100)}" if drop_ratio > 0 else ""
        dataset_train_path = self.database / f'dataset_{dataset}_train{drop_suffix}.pkl'
        dataset_test_path = self.database / f'dataset_{dataset}_test{drop_suffix}.pkl'
        dataset_static_path = self.database / f'dataset_{dataset}_static{drop_suffix}.pkl'

        print(f"train_path: {dataset_train_path}")  # 确认路径
        print(f"static_path: {dataset_static_path}")
        print(f"static_path: {dataset_test_path}")

        # 已有pkl缓存直接加载
        if dataset_static_path.exists() and dataset_train_path.exists():
            if mode == 'train':
                self.dataset_train = torch.load(dataset_train_path)
            self.dataset_test = torch.load(dataset_test_path)
            self.dataset_static = torch.load(dataset_static_path)
            self.static_dataloader()
            return

        if not hasattr(self, 'checkins') or self.checkins is None:
            print("self.checkins 未找到, 正在从文件加载...")
            self.load(self.config.dataset, self.config.dataset_file)

        mapping = self.create_POI_mapping()

        # 80/20划分
        checkins_train, checkins_test = tools.split_user_train_test(self.checkins, 0.8)

        if drop_ratio > 0:
            checkins_train = tools.drop_checkins_global(checkins_train, drop_ratio)

        # Create dataset
        if mode == 'train':
            self.dataset_train = MASHDataset(self.config, checkins_train.copy(), mapping)
            torch.save(self.dataset_train, dataset_train_path)
        self.dataset_test = MASHDataset(self.config, checkins_test.copy(), mapping)
        torch.save(self.dataset_test, dataset_test_path)

        setattr(self.config, 'max_user_num', self.user_count() + 1)
        setattr(self.config, 'max_loc_num', self.location_count() + 1)
        setattr(self.config, 'max_geo_num', self.geohash_count() + 1)
        setattr(self.config, 'max_cat_num', self.category_count() + 1)

        self.dataset_static = torch.LongTensor([
            self.config.max_user_num,
            self.config.max_loc_num,
            self.config.max_geo_num,
            self.config.max_cat_num
        ])

        torch.save(self.dataset_static, dataset_static_path)
        self.static_dataloader()

    def train_dataloader(self):
        return DataLoader(dataset=self.dataset_train,
                          batch_size=self.config.batch_size,
                          pin_memory=True,
                          shuffle=True,
                          follow_batch=['traj_graph', 'geo_graph']
                          )

    def val_dataloader(self):
        return DataLoader(dataset=self.dataset_test,
                          batch_size=self.config.batch_size,
                          shuffle=False,
                          follow_batch=['traj_graph', 'geo_graph']
                          )

    def test_dataloader(self):
        return DataLoader(dataset=self.dataset_test,
                          batch_size=self.config.batch_size,
                          shuffle=False,
                          follow_batch=['traj_graph', 'geo_graph']
                          )

    def static_dataloader(self):
        setattr(self.config, 'max_user_num', int(self.dataset_static[0]))
        setattr(self.config, 'max_loc_num', int(self.dataset_static[1]))
        setattr(self.config, 'max_geo_num', int(self.dataset_static[2]))
        if len(self.dataset_static) > 3:  # 兼容旧的 static 文件
            setattr(self.config, 'max_cat_num', int(self.dataset_static[3]))

    def user_count(self):
        return len(self.checkins['user_id'].unique())

    def location_count(self):
        return len(self.checkins['location_id'].unique())

    def geohash_count(self):
        return len(self.checkins['geohash_id'].unique())

    def category_count(self):
        return len(self.checkins['category_id'].unique())

    def checkins_count(self):
        return len(self.checkins)

    def load_checkins(self, dataset, file):
        if (self.database / f'checkins_{dataset}.pkl').exists():
            checkins = pd.read_pickle(self.database / f'checkins_{dataset}.pkl')
            return checkins

        checkins = pd.read_csv(
            file, sep='\t', names=['user', 'time', 'latitude', 'longitude', 'location', 'category'])
        checkins['time'] = pd.to_datetime(checkins['time'], errors='coerce')
        checkins = checkins.dropna().drop_duplicates()
        checkins.reset_index(drop=True, inplace=True)  # 重置为连续索引
        checkins = tools.geohash_encode(checkins)  # 经纬度转换为geohash
        checkins = self.__convert(checkins)  # 为user，location, geohash创建唯一ID
        checkins.to_pickle(self.database / f'checkins_{dataset}.pkl')  # 缓存到pkl
        return checkins

    def __convert(self, checkins):
        def item2id(checkins, column):
            item = checkins[column].unique()
            item2id = dict(zip(item, range(1, item.size + 1)))
            checkins.insert(checkins.shape[1], f'{column}_id', checkins[column].map(item2id))
            return checkins, item2id

        checkins, user2id = item2id(checkins, 'user')
        checkins, location2id = item2id(checkins, 'location')
        checkins, geohash2id = item2id(checkins, 'geohash')
        checkins, category2id = item2id(checkins, 'category')

        checkins.insert(checkins.shape[1], 'time_unix', checkins['time'].astype('int') // 10**9)

        checkins = checkins.sort_values(['user_id', 'time_unix'])
        checkins.reset_index(drop=True, inplace=True)
        return checkins
