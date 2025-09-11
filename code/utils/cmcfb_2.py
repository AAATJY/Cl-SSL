import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


class AdaptiveMultiScaleRegionContrastiveMemory:
    """
    自适应多尺度区域对比记忆库 (AMR-CMB)
    - 多尺度：为每个尺度的每个类别维护一个有界队列
    - 自适应：支持温度/尺度权重可配置，保留扩展空间
    - 动态内存管理：支持置信度过滤与简单类别平衡策略
    """

    def __init__(
        self,
        feat_dims: Dict[str, int],
        queue_size: int = 200,
        base_tau: float = 0.15,
        device: str = "cuda",
        scale_weights: Optional[Dict[str, float]] = None,
        adaptive_temp: bool = True,
        confidence_threshold: float = 0.75,
        class_balancing: bool = True,
        max_neg_per_class: int = 128
    ):
        """
        Args:
            feat_dims: {scale_name: feat_dim}
            queue_size: 每类队列最大长度
            base_tau: 基础温度
            device: 设备
            scale_weights: 各尺度权重（None 则均分）
            adaptive_temp: 是否预留自适应温度能力（当前以 base_tau 为主）
            confidence_threshold: 置信度阈值（用于可选的伪标注更新）
            class_balancing: 是否启用简易类别平衡（队列满时更可能替换多数类）
            max_neg_per_class: 每类负样本最多采样多少（避免超大内存/时间）
        """
        self.device = device
        self.scales = list(feat_dims.keys())
        self.feat_dims = feat_dims
        self.base_tau = base_tau
        self.adaptive_temp = adaptive_temp
        self.confidence_threshold = confidence_threshold
        self.class_balancing = class_balancing
        self.max_neg_per_class = max_neg_per_class

        # 每个尺度、每个类别一条队列（0 背景，1 前景）
        self.memory: Dict[str, Dict[int, deque]] = {
            s: {0: deque(maxlen=queue_size), 1: deque(maxlen=queue_size)} for s in self.scales
        }
        # 类内计数（用于一个简单的平衡替换策略）
        self.class_counts: Dict[str, Dict[int, int]] = {s: {0: 0, 1: 0} for s in self.scales}

        # 尺度权重
        if scale_weights is None:
            self.scale_weights = {s: 1.0 / len(self.scales) for s in self.scales}
        else:
            self.scale_weights = scale_weights

    # --------------------------
    # 工具函数
    # --------------------------
    def _get_tau(self, scale: str) -> float:
        # 可扩展为“按尺度/难度自适应”，目前返回 base_tau
        return self.base_tau

    def _sample_list(self, scale: str, cls: int) -> Optional[List[torch.Tensor]]:
        dq = self.memory[scale][cls]
        if len(dq) == 0:
            return None
        return [t.to(self.device) for t in list(dq)]

    # --------------------------
    # 记忆库更新
    # --------------------------
    @torch.no_grad()
    def update_from_centers(
        self,
        centers_dict: Dict[str, Dict[int, Optional[torch.Tensor]]],
        confidence_scores: Optional[Dict[str, torch.Tensor]] = None
    ):
        """
        将多尺度的区域中心写入记忆库

        Args:
            centers_dict: {scale: {class_id: tensor([M, C]) or None}}
            confidence_scores: {scale: tensor([M]) in [0,1]} 可选，每个样本的置信度（与 M 对齐）
                               若提供，则仅保留大于阈值的样本进行更新
        """
        for scale, per_class in centers_dict.items():
            for c, centers in per_class.items():
                if centers is None:
                    continue

                centers_to_use = centers
                if confidence_scores is not None and scale in confidence_scores and confidence_scores[scale] is not None:
                    scores = confidence_scores[scale]
                    if scores.dim() == 0:
                        # 单值广播
                        mask = scores.item() > self.confidence_threshold
                    else:
                        mask = scores > self.confidence_threshold
                    if isinstance(mask, bool):
                        if not mask:
                            continue
                    else:
                        centers_to_use = centers[mask]
                        if centers_to_use.shape[0] == 0:
                            continue

                for i in range(centers_to_use.shape[0]):
                    # 类别平衡：若队列已满，可优先替换占比更大的类（此处采用最简单策略：满则popleft）
                    dq = self.memory[scale][int(c)]
                    if len(dq) == dq.maxlen and self.class_balancing:
                        dq.popleft()
                        self.class_counts[scale][int(c)] = max(0, self.class_counts[scale][int(c)] - 1)

                    dq.append(centers_to_use[i].detach().cpu())
                    self.class_counts[scale][int(c)] += 1

    # --------------------------
    # 区域中心计算
    # --------------------------
    @staticmethod
    def compute_multi_scale_region_centers(
        features_dict: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        num_classes: int = 2
    ) -> Dict[str, Dict[int, Optional[torch.Tensor]]]:
        """
        对每个尺度、每个样本、每个类别，计算体素级均值作为区域中心

        Args:
            features_dict: {scale: [B, C, D, H, W]}
            labels: [B, D, H, W]  -> 注意：这里会按各尺度下采样到 feat 的空间大小
        Return:
            {scale: {c: [M, C] 或 None}}
        """
        centers_dict: Dict[str, Dict[int, Optional[torch.Tensor]]] = {}
        B = labels.shape[0]
        for scale, feat in features_dict.items():
            Cf = feat.shape[1]
            Ds, Hs, Ws = feat.shape[-3:]  # 该尺度特征的空间尺寸

            # [FIX] 将标签下采样到该尺度大小（nearest 保持整数类别）
            # labels_resized: [B, Ds, Hs, Ws]
            with torch.no_grad():
                labels_resized = F.interpolate(
                    labels.float().unsqueeze(1),  # [B,1,D,H,W]
                    size=(Ds, Hs, Ws),
                    mode='nearest'
                ).squeeze(1).long()

            scale_centers: Dict[int, Optional[torch.Tensor]] = {}
            for c in range(num_classes):
                per_sample: List[torch.Tensor] = []
                for b in range(B):
                    mask = (labels_resized[b] == c)  # [Ds,Hs,Ws]
                    if mask.sum() == 0:
                        continue
                    feat_b = feat[b]  # [C,Ds,Hs,Ws]
                    sel = feat_b.view(Cf, -1)[:, mask.view(-1).bool()]  # [C, N_c]
                    if sel.shape[-1] == 0:
                        continue
                    center = sel.mean(dim=1)  # [C]
                    per_sample.append(center.unsqueeze(0))
                if len(per_sample) == 0:
                    scale_centers[c] = None
                else:
                    scale_centers[c] = torch.cat(per_sample, dim=0)  # [M,C]
            centers_dict[scale] = scale_centers
        return centers_dict

    # --------------------------
    # 对比损失
    # --------------------------
    def _contrastive_loss_one_scale(
        self,
        scale: str,
        centers_per_class: Dict[int, Optional[torch.Tensor]]
    ) -> Tuple[torch.Tensor, Dict[int, float]]:
        """
        在单个尺度上计算 InfoNCE 损失（二分类）
        Return:
            loss: 标量
            stats: {class_id: loss_value or 0.0}
        """
        device = self.device
        tau = self._get_tau(scale)
        total_loss = torch.tensor(0.0, device=device)
        total_cnt = 0
        per_class_loss = {0: 0.0, 1: 0.0}
        per_class_cnt = {0: 0, 1: 0}

        for c in [0, 1]:
            anchors = centers_per_class.get(c, None)
            if anchors is None:
                continue
            if anchors.dim() == 1:
                anchors = anchors.unsqueeze(0)  # [1,C]
            anchors = anchors.to(device)

            pos_list = self._sample_list(scale, c)
            neg_list = self._sample_list(scale, 1 - c)

            if pos_list is None or len(pos_list) == 0 or neg_list is None or len(neg_list) == 0:
                continue

            # 限制负样本规模（每类最多 self.max_neg_per_class）
            if self.max_neg_per_class is not None and len(neg_list) > self.max_neg_per_class:
                # 随机下采样（可替换为困难样本优先策略）
                idx = torch.randperm(len(neg_list))[: self.max_neg_per_class].tolist()
                neg_list = [neg_list[i] for i in idx]

            pos = torch.stack(pos_list).to(device)  # [P,C]
            neg = torch.stack(neg_list).to(device)  # [N,C]
            pos = F.normalize(pos, dim=1)
            neg = F.normalize(neg, dim=1)

            for i in range(anchors.shape[0]):
                anchor = F.normalize(anchors[i], dim=0)
                pos_sim = torch.exp(torch.matmul(pos, anchor) / tau).sum()          # 标量
                neg_sim = torch.exp(torch.matmul(neg, anchor) / tau).sum()          # 标量
                loss_i = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8))
                total_loss += loss_i
                total_cnt += 1
                per_class_loss[c] += float(loss_i.detach().cpu())
                per_class_cnt[c] += 1

        if total_cnt == 0:
            return torch.tensor(0.0, device=device), {k: 0.0 for k in per_class_loss}
        # 平均
        avg = total_loss / total_cnt
        stats = {k: (per_class_loss[k] / per_class_cnt[k] if per_class_cnt[k] > 0 else 0.0) for k in per_class_loss}
        return avg, stats

    def contrastive_loss_all_scales(
        self,
        centers_dict: Dict[str, Dict[int, Optional[torch.Tensor]]]
    ) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, Dict[int, float]]]:
        """
        汇总多尺度对比损失
        Return:
            total_loss: 加权总损失
            per_scale_loss: {scale: loss_value}
            per_scale_class_loss: {scale: {class_id: loss_value}}
        """
        device = self.device
        total = torch.tensor(0.0, device=device)
        per_scale_loss: Dict[str, float] = {}
        per_scale_class_loss: Dict[str, Dict[int, float]] = {}

        for scale, centers in centers_dict.items():
            loss_s, stats_s = self._contrastive_loss_one_scale(scale, centers)
            w = self.scale_weights.get(scale, 1.0 / max(1, len(self.scale_weights)))
            total = total + w * loss_s
            per_scale_loss[scale] = float(loss_s.detach().cpu())
            per_scale_class_loss[scale] = stats_s

        return total, per_scale_loss, per_scale_class_loss