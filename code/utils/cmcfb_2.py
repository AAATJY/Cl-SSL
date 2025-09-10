from collections import deque
import torch
import torch.nn.functional as F
import numpy as np


class AdaptiveMultiScaleRegionContrastiveMemory:
    def __init__(self, feat_dims, queue_size=200, base_tau=0.1,
                 device="cuda", scale_weights=None, adaptive_temp=True,
                 confidence_threshold=0.7, class_balancing=True):
        """
        自适应多尺度区域对比记忆库 (AMR-CMB)

        Args:
            feat_dims: 字典，键为尺度名称，值为特征维度
            queue_size: 每类队列最大长度
            base_tau: 基础温度参数
            device: 设备
            scale_weights: 不同尺度的权重，如果为None则自动学习
            adaptive_temp: 是否使用自适应温度
            confidence_threshold: 置信度阈值，用于筛选高质量样本
            class_balancing: 是否使用类别平衡
        """
        self.base_tau = base_tau
        self.device = device
        self.feat_dims = feat_dims
        self.scales = list(feat_dims.keys())
        self.confidence_threshold = confidence_threshold
        self.class_balancing = class_balancing

        # 初始化每个尺度的记忆库
        self.memory = {
            scale: {
                0: deque(maxlen=queue_size),
                1: deque(maxlen=queue_size)
            } for scale in self.scales
        }

        # 尺度权重（可学习或固定）
        if scale_weights is None:
            self.scale_weights = {scale: 1.0 / len(self.scales) for scale in self.scales}
            self.learnable_weights = True
        else:
            self.scale_weights = scale_weights
            self.learnable_weights = False

        # 自适应温度参数
        self.adaptive_temp = adaptive_temp
        self.tau_factor = 1.0

        # 类别统计
        self.class_counts = {0: 0, 1: 0}

    def update_tau_factor(self, iteration, max_iterations):
        """根据训练进度调整温度因子"""
        if self.adaptive_temp:
            # 早期训练使用较高温度（更平滑的分布），后期使用较低温度（更尖锐的分布）
            self.tau_factor = 0.5 + 0.5 * (1 - iteration / max_iterations)

    def update_scale_weights(self, scale_losses):
        """根据各尺度的损失更新权重"""
        if self.learnable_weights:
            total_loss = sum(scale_losses.values())
            for scale in self.scales:
                # 损失越小的尺度权重越大
                self.scale_weights[scale] = 1.0 - (scale_losses[scale] / total_loss)

            # 归一化权重
            weight_sum = sum(self.scale_weights.values())
            for scale in self.scales:
                self.scale_weights[scale] /= weight_sum

    @torch.no_grad()
    def update_from_centers(self, centers_dict, scale, confidence_scores=None):
        """
        根据置信度筛选高质量样本更新记忆库

        Args:
            centers_dict: {class_id: tensor([B, C])} 区域中心
            scale: 尺度名称
            confidence_scores: 每个样本的置信度，用于筛选高质量样本
        """
        for c, centers in centers_dict.items():
            if centers is None:
                continue

            # 如果有置信度分数，筛选高质量样本
            if confidence_scores is not None:
                mask = confidence_scores > self.confidence_threshold
                centers = centers[mask]
                if centers.shape[0] == 0:
                    continue

            # 更新记忆库
            for i in range(centers.shape[0]):
                # 类别平衡：如果某类样本过多，随机替换而不是总是添加
                if self.class_balancing and len(self.memory[scale][c]) >= self.memory[scale][c].maxlen:
                    if np.random.rand() > self.class_counts[c] / sum(self.class_counts.values()):
                        self.memory[scale][c].popleft()
                        self.class_counts[c] -= 1
                    else:
                        continue

                self.memory[scale][c].append(centers[i].detach().cpu())
                self.class_counts[c] += 1

    def sample_with_priority(self, scale, cls, num_samples):
        """优先级采样：更频繁地采样难以区分的样本"""
        samples = list(self.memory[scale][cls])
        if len(samples) == 0:
            return None

        # 计算样本间的相似度作为难度指标
        if len(samples) > 1:
            sample_tensors = torch.stack(samples).to(self.device)
            similarities = F.cosine_similarity(
                sample_tensors.unsqueeze(1),
                sample_tensors.unsqueeze(0),
                dim=2
            )
            # 难度 = 1 - 与同类样本的平均相似度 + 与异类样本的平均相似度
            difficulties = []
            for i in range(len(samples)):
                same_class_mask = torch.ones(len(samples), dtype=bool)
                same_class_mask[i] = False
                same_class_sim = similarities[i, same_class_mask].mean()

                difficulties.append(1 - same_class_sim)

            difficulties = torch.tensor(difficulties)
            probabilities = F.softmax(difficulties, dim=0)
            indices = torch.multinomial(probabilities, min(num_samples, len(samples)), replacement=False)
            return [samples[i] for i in indices.cpu().numpy()]
        else:
            return samples

    def contrastive_loss(self, anchors_dict, scale):
        """
        计算对比损失，使用自适应温度和优先级采样

        Args:
            anchors_dict: {class_id: tensor([B, C])} 锚点特征
            scale: 尺度名称

        Returns:
            loss: 对比损失
            num_valid: 有效样本数
        """
        device = self.device
        total_loss = torch.tensor(0.0, device=device)
        num_valid = 0

        # 自适应温度
        tau = self.base_tau * self.tau_factor

        for c, anchors in anchors_dict.items():
            if anchors is None:
                continue

            # 获取正负样本（使用优先级采样）
            pos_samples = self.sample_with_priority(scale, c, 10)  # 采样10个正样本
            neg_samples = []
            for other_c in [0, 1]:
                if other_c != c:
                    samples = self.sample_with_priority(scale, other_c, 5)  # 每个负类采样5个样本
                    if samples is not None:
                        neg_samples.extend(samples)

            if pos_samples is None or len(pos_samples) == 0 or len(neg_samples) == 0:
                continue

            # 转换为张量
            pos_samples = torch.stack(pos_samples).to(device)
            neg_samples = torch.stack(neg_samples).to(device)

            # 计算每个锚点的损失
            for i in range(anchors.shape[0]):
                anchor = F.normalize(anchors[i], dim=0)
                pos_norm = F.normalize(pos_samples, dim=1)
                neg_norm = F.normalize(neg_samples, dim=1)

                pos_sim = torch.exp(torch.matmul(anchor.unsqueeze(0), pos_norm.t()) / tau).sum()
                neg_sim = torch.exp(torch.matmul(anchor.unsqueeze(0), neg_norm.t()) / tau).sum()

                loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8))
                total_loss += loss
                num_valid += 1

        return total_loss, num_valid

    def multi_scale_contrastive_loss(self, multi_scale_anchors):
        """
        计算多尺度对比损失

        Args:
            multi_scale_anchors: 字典 {scale: {class_id: tensor}} 多尺度锚点

        Returns:
            total_loss: 总对比损失
            scale_losses: 各尺度的损失（用于更新权重）
        """
        total_loss = torch.tensor(0.0, device=self.device)
        scale_losses = {}

        for scale, anchors_dict in multi_scale_anchors.items():
            if scale not in self.scales:
                continue

            loss, num_valid = self.contrastive_loss(anchors_dict, scale)
            if num_valid > 0:
                scale_loss = loss / num_valid
                total_loss += self.scale_weights[scale] * scale_loss
                scale_losses[scale] = scale_loss.item()

        return total_loss, scale_losses

    @staticmethod
    def compute_multi_scale_region_centers(features_dict, labels, num_classes=2):
        """
        从多尺度特征中计算区域中心

        Args:
            features_dict: 字典 {scale_name: tensor([B, C, D, H, W])} 多尺度特征
            labels: 标签张量 [B, D, H, W]
            num_classes: 类别数

        Returns:
            centers_dict: 字典 {scale: {class_id: tensor}} 多尺度区域中心
        """
        centers_dict = {}

        for scale, features in features_dict.items():
            B, Cf, D, H, W = features.shape
            scale_centers = {}

            for c in range(num_classes):
                per_sample_centers = []
                for b in range(B):
                    mask = (labels[b] == c)
                    if mask.sum() == 0:
                        continue

                    feat = features[b]
                    feat_flat = feat.view(Cf, -1)
                    mask_flat = mask.view(-1).bool()
                    sel = feat_flat[:, mask_flat]

                    if sel.shape[-1] == 0:
                        continue

                    center = sel.mean(dim=1)
                    per_sample_centers.append(center.unsqueeze(0))

                if len(per_sample_centers) == 0:
                    scale_centers[c] = None
                else:
                    scale_centers[c] = torch.cat(per_sample_centers, dim=0)

            centers_dict[scale] = scale_centers

        return centers_dict