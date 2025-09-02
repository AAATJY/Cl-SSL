from collections import deque

import torch
import torch.nn.functional as F


# ----------------------------
# CFCMB（二分类，队列机制）
# ----------------------------
class RegionContrastiveMemoryBinary:
    def __init__(self, feat_dim, queue_size=200, tau=0.1, device="cuda"):
        """
        二分类 CFCMB（0 背景，1 前景）
        feat_dim: 特征通道数（student_features 通道数）
        queue_size: 每类队列最大长度
        tau: 温度
        """
        self.tau = tau
        self.device = device
        self.feat_dim = feat_dim
        self.memory = {
            0: deque(maxlen=queue_size),
            1: deque(maxlen=queue_size)
        }

    @torch.no_grad()
    def update_from_centers(self, centers_dict):
        """
        centers_dict: {class_id: tensor([C])} 或 {class_id: tensor([B, C])}
        将 centers 写入对应队列（detach、cpu 存放）
        """
        for c, v in centers_dict.items():
            # v could be shape [C] or [B, C]
            if v is None:
                continue
            if v.dim() == 1:
                self.memory[int(c)].append(v.detach().cpu())
            else:
                for i in range(v.shape[0]):
                    self.memory[int(c)].append(v[i].detach().cpu())

    def sample_list(self, cls):
        """返回 list of tensors 在 GPU（或 None）"""
        if len(self.memory[cls]) == 0:
            return None
        return [t.to(self.device) for t in list(self.memory[cls])]

    def contrastive_loss_for_centers(self, centers_dict):
        """
        centers_dict: {class_id: tensor([B, C]) or tensor([C])}
        return: scalar loss (torch.Tensor)
        """
        device = self.device
        total_loss = torch.tensor(0.0, device=device)
        count = 0
        for c, q in centers_dict.items():
            if q is None:
                continue
            # normalize
            if q.dim() == 1:
                q_list = [q.to(device)]
            else:
                q_list = [q[i].to(device) for i in range(q.shape[0])]
            pos_list = self.sample_list(c)
            neg_list = []
            for k in [0, 1]:
                if k == c:
                    continue
                lst = self.sample_list(k)
                if lst is not None:
                    neg_list.extend(lst)

            # if no pos or no neg -> skip
            if (pos_list is None) or (len(pos_list) == 0) or (len(neg_list) == 0):
                continue

            for anchor in q_list:
                anchor = F.normalize(anchor, dim=0)
                pos_norm = [F.normalize(p.to(device), dim=0) for p in pos_list]
                neg_norm = [F.normalize(n.to(device), dim=0) for n in neg_list]
                pos_sim = torch.stack([torch.exp(torch.dot(anchor, p) / self.tau) for p in pos_norm]).sum()
                neg_sim = torch.stack([torch.exp(torch.dot(anchor, n) / self.tau) for n in neg_norm]).sum()
                loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8))
                total_loss += loss
                count += 1
        if count == 0:
            return torch.tensor(0.0, device=device)
        return total_loss / count

    def compute_region_centers_3d(features, labels, num_classes=2):
        """
        features: [B, C, D, H, W]
        labels: [B, D, H, W]
        return: dict {class_id: tensor [B, C] or None}
        """
        B, Cf, D, H, W = features.shape
        centers = {}
        for c in range(num_classes):
            # for each sample in batch, compute center if exists
            per_sample_centers = []
            for b in range(B):
                mask = (labels[b] == c)  # [D, H, W]
                if mask.sum() == 0:
                    continue
                feat = features[b]  # [C, D, H, W]
                feat_flat = feat.view(Cf, -1)  # [C, N]
                mask_flat = mask.view(-1).bool()
                sel = feat_flat[:, mask_flat]  # [C, N_c]
                if sel.shape[-1] == 0:
                    continue
                center = sel.mean(dim=1)  # [C]
                per_sample_centers.append(center.unsqueeze(0))  # [1, C]
            if len(per_sample_centers) == 0:
                centers[c] = None
            else:
                centers[c] = torch.cat(per_sample_centers, dim=0)  # [M, C]
        return centers