import torch
import torch.nn.functional as F

def nt_xent_loss(feat_q, feat_k, temperature=0.2):
    feat_q = F.normalize(feat_q, dim=1)
    feat_k = F.normalize(feat_k, dim=1)
    N = feat_q.shape[0]
    logits = torch.mm(feat_q, feat_k.t()) / temperature
    labels = torch.arange(N).long().to(feat_q.device)
    loss = F.cross_entropy(logits, labels)
    return loss

def voxel_contrastive_loss_uncertainty(
    feat_map, edge_mask, core_mask, uncertainty=None,
    temperature=0.2, max_samples=1024, hard_negative=True):
    """
    feat_map: [B, C, D, H, W]
    edge_mask/core_mask: [B, 1, D, H, W]
    uncertainty: [B, 1, D, H, W] (optional)
    体素级对比损失，正样本来自核心/边缘mask区域，可以结合不确定性进一步采样
    """
    B, C, D, H, W = feat_map.shape
    # 1. 提取核心正样本/边缘正样本
    feature_flat = feat_map.permute(0,2,3,4,1).reshape(-1, C)
    edge_flat = edge_mask.reshape(-1).bool()
    core_flat = core_mask.reshape(-1).bool()
    pos_idx = (edge_flat | core_flat).nonzero(as_tuple=True)[0]
    if pos_idx.numel() < 2:
        return torch.tensor(0.0, device=feat_map.device, requires_grad=True)
    if pos_idx.numel() > max_samples:
        perm_idx = torch.randperm(pos_idx.numel(), device=feat_map.device)[:max_samples]
        pos_idx = pos_idx[perm_idx]
    pos_feats = feature_flat[pos_idx]

    # 2. （可选）不确定性加权采样正样本：例如 uncertainty<阈值
    if uncertainty is not None:
        uncertainty_flat = uncertainty.reshape(-1)
        # threshold = uncertainty_flat.median() # e.g. use median
        low_uncertainty = uncertainty_flat[pos_idx].topk(max_samples, largest=False).indices
        pos_feats = pos_feats[low_uncertainty]

    # 3. 负样本采样：选取batch其他非mask区域
    neg_idx = (~(edge_flat | core_flat)).nonzero(as_tuple=True)[0]
    if neg_idx.numel() > max_samples:
        perm_idx = torch.randperm(neg_idx.numel(), device=feat_map.device)[:max_samples]
        neg_idx = neg_idx[perm_idx]
    neg_feats = feature_flat[neg_idx]

    # 4. 构造anchor-positive对 & hard negative
    N = pos_feats.size(0)
    if N < 2:
        return torch.tensor(0.0, device=feat_map.device)
    perm = torch.randperm(N, device=feat_map.device)
    feat_q = pos_feats
    feat_k = pos_feats[perm]

    # 5. Hard negative mining
    if hard_negative and neg_feats.size(0) > 0:
        # 计算 anchor 和所有 neg_feats 的相似度，取最难的neg
        feat_q_norm = F.normalize(feat_q, dim=1)
        neg_norm = F.normalize(neg_feats, dim=1)
        sim = torch.mm(feat_q_norm, neg_norm.t())  # [N, N_neg]
        hard_neg_idx = sim.argmax(dim=1) # for each anchor, pick hardest negative
        feat_n = neg_feats[hard_neg_idx]
        # anchor-positive, anchor-hardnegative, NT-Xent
        pos_sim = (feat_q * feat_k).sum(dim=1) / temperature
        neg_sim = (feat_q * feat_n).sum(dim=1) / temperature
        logits = torch.stack([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(N, dtype=torch.long, device=feat_map.device)
        loss = F.cross_entropy(logits, labels)
        return loss
    else:
        return nt_xent_loss(feat_q, feat_k, temperature)

def patch_contrastive_loss(feat_map, mask, patch_size=(8,8,8), temperature=0.2, max_patches=256):
    B, C, D, H, W = feat_map.shape
    stride = patch_size
    pool = torch.nn.AvgPool3d(kernel_size=patch_size, stride=patch_size)
    mask_pool = pool(mask.float())
    feat_pool = pool(feat_map * mask.float())
    # mask_pool>0.5为有效patch
    valid = (mask_pool.squeeze(1) > 0.5)
    feat_patches = feat_pool.permute(0,2,3,4,1)[valid].reshape(-1, C)
    if feat_patches.size(0) < 2:
        return torch.tensor(0.0, device=feat_map.device)
    if feat_patches.size(0) > max_patches:
        perm = torch.randperm(feat_patches.size(0))[:max_patches]
        feat_patches = feat_patches[perm]
    # shuffle for anchor-pos
    perm = torch.randperm(feat_patches.size(0))
    feat_q = feat_patches
    feat_k = feat_patches[perm]
    return nt_xent_loss(feat_q, feat_k, temperature)