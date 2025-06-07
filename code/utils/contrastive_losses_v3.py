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
    feature_flat = feat_map.permute(0,2,3,4,1).reshape(-1, C)
    edge_flat = edge_mask.reshape(-1).bool()
    core_flat = core_mask.reshape(-1).bool()
    pos_idx = (edge_flat | core_flat).nonzero(as_tuple=True)[0]
    if pos_idx.numel() < 2:
        return torch.tensor(0.0, device=feat_map.device, requires_grad=True)

    # === 不确定性加权采样正样本 ===
    if uncertainty is not None:
        uncertainty_flat = uncertainty.reshape(-1)
        valid_uncertainty = uncertainty_flat[pos_idx]
        num_valid = valid_uncertainty.numel()
        k = min(max_samples, num_valid)
        if k < 2:
            return torch.tensor(0.0, device=feat_map.device, requires_grad=True)
        # 采样低不确定性体素（也可采高不确定性体素用于分析，按需修改）
        low_uncertainty_indices = valid_uncertainty.topk(k, largest=False).indices
        pos_feats = feature_flat[pos_idx][low_uncertainty_indices]
    else:
        num_valid = pos_idx.numel()
        k = min(max_samples, num_valid)
        if k < 2:
            return torch.tensor(0.0, device=feat_map.device, requires_grad=True)
        perm_idx = torch.randperm(num_valid, device=feat_map.device)[:k]
        pos_feats = feature_flat[pos_idx][perm_idx]

    # === 负样本采样 ===
    neg_idx = (~(edge_flat | core_flat)).nonzero(as_tuple=True)[0]
    if neg_idx.numel() > max_samples:
        perm_idx = torch.randperm(neg_idx.numel(), device=feat_map.device)[:max_samples]
        neg_idx = neg_idx[perm_idx]
    neg_feats = feature_flat[neg_idx]

    # === 构造anchor-positive对 & hard negative ===
    N = pos_feats.size(0)
    if N < 2:
        return torch.tensor(0.0, device=feat_map.device, requires_grad=True)
    perm = torch.randperm(N, device=feat_map.device)
    feat_q = pos_feats
    feat_k = pos_feats[perm]

    # === Hard negative mining ===
    if hard_negative and neg_feats.size(0) > 0:
        feat_q_norm = F.normalize(feat_q, dim=1)
        neg_norm = F.normalize(neg_feats, dim=1)
        sim = torch.mm(feat_q_norm, neg_norm.t())  # [N, N_neg]
        hard_neg_idx = sim.argmax(dim=1) # for each anchor, pick hardest negative
        feat_n = neg_feats[hard_neg_idx]
        pos_sim = (feat_q * feat_k).sum(dim=1) / temperature
        neg_sim = (feat_q * feat_n).sum(dim=1) / temperature
        logits = torch.stack([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(N, dtype=torch.long, device=feat_map.device)
        loss = F.cross_entropy(logits, labels)
        return loss
    else:
        return nt_xent_loss(feat_q, feat_k, temperature)

def patch_contrastive_loss_dycon(
    feat_map,
    mask,
    uncertainty=None,
    patch_size=(8,8,8),
    temperature=0.2,
    max_patches=256,
    hard_negative=True,
    dynamic_threshold=True,
    epoch=None,
    max_epoch=100
):
    """
    DyCON式补丁级对比损失。
    - feat_map: [B, C, D, H, W]
    - mask: [B, 1, D, H, W]，区域mask，建议用core_mask（置信度高的区域）
    - uncertainty: [B, 1, D, H, W]（可选），用于过滤高不确定patch
    - dynamic_threshold: 是否动态过滤uncertainty
    - hard_negative: 是否启用hard negative mining
    """

    B, C, D, H, W = feat_map.shape

    # 1. 区域掩码结合uncertainty进一步过滤
    if uncertainty is not None and dynamic_threshold:
        # 动态阈值，可根据epoch进行rampup
        if epoch is not None:
            ramp = min(epoch / max_epoch, 1.0)
            uncertainty_thr = uncertainty.mean() * (0.5 + 0.5 * ramp)
        else:
            uncertainty_thr = uncertainty.mean()
        mask = mask * (uncertainty < uncertainty_thr).float()  # 只用低uncertainty区域

    # 2. patch pooling
    pool = torch.nn.AvgPool3d(kernel_size=patch_size, stride=patch_size)
    mask_pool = pool(mask.float())
    feat_pool = pool(feat_map * mask.float())
    valid = (mask_pool.squeeze(1) > 0.5)
    feat_patches = feat_pool.permute(0,2,3,4,1)[valid].reshape(-1, C)
    if feat_patches.size(0) < 2:
        return torch.tensor(0.0, device=feat_map.device, requires_grad=True)
    # 3. uncertainty优先采样
    if uncertainty is not None:
        uncertainty_patches = pool(uncertainty.float())[valid].reshape(-1)
        k = min(max_patches, feat_patches.size(0))
        idx = torch.topk(-uncertainty_patches, k).indices # 低uncertainty优先
        feat_patches = feat_patches[idx]
    else:
        if feat_patches.size(0) > max_patches:
            perm = torch.randperm(feat_patches.size(0), device=feat_map.device)[:max_patches]
            feat_patches = feat_patches[perm]

    # 4. hard negative mining
    perm = torch.randperm(feat_patches.size(0), device=feat_map.device)
    feat_q = feat_patches
    feat_k = feat_patches[perm]
    if hard_negative and feat_patches.size(0) > 2:
        # 计算相似度，选出hard negative
        feat_q_norm = F.normalize(feat_q, dim=1)
        sim = torch.mm(feat_q_norm, feat_q_norm.t())
        # 排除自身（对角线），选最像的其他patch为hard negative
        sim.fill_diagonal_(-1.0)
        hard_neg_idx = sim.argmax(dim=1)
        feat_n = feat_patches[hard_neg_idx]
        pos_sim = (feat_q * feat_k).sum(dim=1) / temperature
        neg_sim = (feat_q * feat_n).sum(dim=1) / temperature
        logits = torch.stack([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(feat_q.size(0), dtype=torch.long, device=feat_q.device)
        loss = F.cross_entropy(logits, labels)
        return loss
    else:
        return nt_xent_loss(feat_q, feat_k, temperature)