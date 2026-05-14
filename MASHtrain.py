from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.optim.lr_scheduler import LinearLR, SequentialLR, StepLR
from tools import *


class Trainer:
    def __init__(self, config, logger=None, gpu=0):
        self.config = config
        self.logger = logger if logger is not None else init_logger()

        if gpu == -1:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(
                f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

        self.model_dir = Path(self.config.model_dir)
        if not self.model_dir.exists():
            self.model_dir.mkdir()

    def compute_bpr_loss(self, pred, label, mask, num_neg=1):
        B, L, N = pred.shape
        device = pred.device

        # 生成负样本索引，初始化为 0
        neg_ids = torch.zeros(B, L, dtype=torch.long, device=device)

        for b in range(B):
            # 当前 batch 的候选集：mask 为 False 的位置才是候选
            candidates = (~mask[b]).nonzero().squeeze(1)
            if len(candidates) == 0:
                continue
            for t in range(L):
                pos = label[b, t]
                if pos == 0:
                    continue
                # 从候选集中排除正样本
                possible = candidates[candidates != pos]
                if len(possible) > 0:
                    neg_ids[b, t] = possible[torch.randint(0, len(possible), (1,))]

        # 获取正负样本得分
        pos_scores = pred.gather(2, label.unsqueeze(2)).squeeze(2)  # [B, L]
        neg_scores = pred.gather(2, neg_ids.unsqueeze(2)).squeeze(2)  # [B, L]

        # 有效位置：标签非零且负样本非零
        valid = (label != 0) & (neg_ids != 0)
        diff = pos_scores - neg_scores
        loss = -torch.log(torch.sigmoid(diff) + 1e-8) * valid.float()
        return loss.sum() / (valid.sum() + 1e-8)

    def train(self, model, dataloader):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        train_dl = dataloader.train_dataloader()
        val_dl = dataloader.val_dataloader()
        model = model.to(self.device)

        self.logger.info('start training...')
        for epoch in range(self.config.epochs):
            self._train_epoch(epoch, model, train_dl, optimizer, scheduler, criterion)

            if (epoch + 1) % 1 == 0:
                self._val_epoch(model, val_dl, criterion)

            model_path = self.model_dir / f"model_{epoch+1}.pkl"
            torch.save(model.state_dict(), model_path)
        self.logger.info('training done!')

    def _train_epoch(self, epoch, model, train_dl, optimizer, scheduler, criterion):
        model.train()
        train_loss = []
        pbar = tqdm(train_dl, total=len(train_dl))
        for idx, dl in enumerate(pbar):
            user, traj, geo, time, traj_cat, center_traj, long_traj, long_traj_cat, dt, label_traj, \
                label_geo, label_cat, negihbors_mask, traj_graph, geo_graph, time_feat, hour = dl
            user = user.to(self.device)
            traj = traj.to(self.device)
            geo = geo.to(self.device)
            time = time.to(self.device)
            traj_cat = traj_cat.to(self.device)
            center_traj = center_traj.to(self.device)
            long_traj = long_traj.to(self.device)
            long_traj_cat = long_traj_cat.to(self.device)
            dt = dt.to(self.device)
            label_traj = label_traj.to(self.device)
            label_geo = label_geo.to(self.device)
            label_cat = label_cat.to(self.device)
            negihbors_mask = negihbors_mask.to(self.device)
            traj_graph = traj_graph.to(self.device)
            geo_graph = geo_graph.to(self.device)
            time_feat = time_feat.to(self.device)
            hour = hour.to(self.device)

            optimizer.zero_grad()
            pred = model(user, traj, geo, time, traj_cat, center_traj, long_traj, long_traj_cat, dt, traj_graph,
                         geo_graph, time_feat, hour)  # 前向传播
            is_pred_tuple = False
            # 如果模型有多个输出（POI、区域、类别）
            if isinstance(pred, tuple):
                pred, pred_geo, pred_cat = pred
                is_pred_tuple = True
            # 应用负采样掩码
            if self.config.mask:
                negihbors_mask = negihbors_mask.unsqueeze(1).repeat(
                    1, self.config.max_sequence_length, 1)
                # 屏蔽非候选POI
                pred.masked_fill_(negihbors_mask, -1000)

            loss_all = criterion(pred.permute(0, 2, 1), label_traj)
            if is_pred_tuple:
                loss_geo = criterion(pred_geo.permute(0, 2, 1), label_geo)
                loss_cat = criterion(pred_cat.permute(0, 2, 1), label_cat)
                loss_all = loss_all + loss_geo * self.config.loss_rate + loss_cat * 0.5
                bpr_loss = self.compute_bpr_loss(pred, label_traj, negihbors_mask)
                loss_all = loss_all + 0.1 * bpr_loss

            # 反向传播
            loss_all.backward()
            optimizer.step()
            train_loss.append(loss_all.item())

            pbar.set_description(f'Epoch [{epoch + 1}/{self.config.epochs}]')
            pbar.set_postfix(loss=np.mean(train_loss), lr=scheduler.get_last_lr()[0])

        scheduler.step()

    @torch.no_grad()
    def _val_epoch(self, model, val_dl, criterion):
        model.eval()
        val_loss, val_acc = [], []
        vbar = tqdm(val_dl, desc='valid', total=len(val_dl))
        for idx, dl in enumerate(vbar):
            user, traj, geo, time, traj_cat, center_traj, long_traj, long_traj_cat, dt, label_traj, \
                label_geo, label_cat, negihbors_mask, traj_graph, geo_graph, time_feat, hour = dl
            user = user.to(self.device)
            traj = traj.to(self.device)
            geo = geo.to(self.device)
            time = time.to(self.device)
            traj_cat = traj_cat.to(self.device)
            center_traj = center_traj.to(self.device)
            long_traj = long_traj.to(self.device)
            long_traj_cat = long_traj_cat.to(self.device)
            dt = dt.to(self.device)
            label_traj = label_traj.to(self.device)
            label_geo = label_geo.to(self.device)
            label_cat = label_cat.to(self.device)
            negihbors_mask = negihbors_mask.to(self.device)
            traj_graph = traj_graph.to(self.device)
            geo_graph = geo_graph.to(self.device)
            time_feat = time_feat.to(self.device)
            hour = hour.to(self.device)

            pred = model(user, traj, geo, time, traj_cat, center_traj, long_traj, long_traj_cat, dt, traj_graph,
                         geo_graph, time_feat, hour)
            is_pred_tuple = False
            if isinstance(pred, tuple):
                pred, pred_geo, pred_cat = pred
                is_pred_tuple = True
            if self.config.mask:
                negihbors_mask = negihbors_mask.unsqueeze(1).repeat(
                    1, self.config.max_sequence_length, 1)
                pred.masked_fill_(negihbors_mask, -1000)

            loss_all = criterion(pred.permute(0, 2, 1), label_traj)
            if is_pred_tuple:
                loss_geo = criterion(pred_geo.permute(0, 2, 1), label_geo)
                loss_cat = criterion(pred_cat.permute(0, 2, 1), label_cat)
                loss_all = loss_all + loss_geo * self.config.loss_rate + loss_cat * 0.5

            val_acc.append(calculate_acc(pred, label_traj))
            val_loss.append(loss_all.item())

            mean_acc = torch.concat(val_acc, dim=1).mean(dim=1).cpu().tolist()
            mean_acc = [round(acc, 4) for acc in mean_acc]
            vbar.set_postfix(val_loss=f'{np.mean(val_loss):.4f}', acc=mean_acc)

    @torch.no_grad()
    def test(self, model, dataloader, model_path):
        # prepare dataloader
        test_dl = dataloader.test_dataloader()

        model = model.to(self.device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        test_acc = []
        tbar = tqdm(test_dl, desc='test', total=len(test_dl))
        self.logger.info('start testing...')
        for idx, dl in enumerate(tbar):
            user, traj, geo, time, traj_cat, center_traj, long_traj, long_traj_cat, dt, label_traj, \
                label_geo, negihbors_mask, traj_graph, geo_graph = dl
            user = user.to(self.device)
            traj = traj.to(self.device)
            geo = geo.to(self.device)
            time = time.to(self.device)
            traj_cat = traj_cat.to(self.device)
            center_traj = center_traj.to(self.device)
            long_traj = long_traj.to(self.device)
            long_traj_cat = long_traj_cat.to(self.device)
            dt = dt.to(self.device)
            label_traj = label_traj.to(self.device)
            label_geo = label_geo.to(self.device)
            negihbors_mask = negihbors_mask.to(self.device)
            traj_graph = traj_graph.to(self.device)
            geo_graph = geo_graph.to(self.device)

            pred = model(user, traj, geo, time, traj_cat, center_traj, long_traj, long_traj_cat, dt, traj_graph,
                         geo_graph)
            if isinstance(pred, tuple):
                pred, pred_geo = pred
            if self.config.mask:
                negihbors_mask = negihbors_mask.unsqueeze(1).repeat(
                    1, self.config.max_sequence_length, 1)
                pred.masked_fill_(negihbors_mask, -1000)

            test_acc.append(calculate_acc(pred, label_traj))
            # update pbar
            mean_acc = torch.concat(test_acc, dim=1).mean(dim=1).cpu().tolist()
            mean_acc = [round(acc, 4) for acc in mean_acc]
            tbar.set_postfix(acc=mean_acc)

        self.logger.info('test done.')
        self.logger.info('-------------------------------------')
        self.logger.info('test result:')
        self.logger.info(f'Acc@1: {mean_acc[0]}')
        self.logger.info(f'Acc@5: {mean_acc[1]}')
        self.logger.info(f'Acc@10: {mean_acc[2]}')
        self.logger.info(f'Acc@20: {mean_acc[3]}')
        self.logger.info(f'MRR: {mean_acc[4]}')
