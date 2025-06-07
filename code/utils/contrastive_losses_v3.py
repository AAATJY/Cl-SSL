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

def voxel_contrastive_loss(feat_map, mask, temperature=0.2, max_samples=2048):
    B, C, D, H, W = feat_map.shape
    mask = mask.to(dtype=torch.bool)
    if mask.shape != (B, 1, D, H, W):
        raise ValueError(f"mask shape {mask.shape} incompatible with feat_map {feat_map.shape}")
    feat_flat = feat_map.permute(0,2,3,4,1).reshape(-1, C)
    mask_flat = mask.reshape(-1)
    pos_idx = mask_flat.nonzero(as_tuple=True)[0]
    if pos_idx.numel() < 2:
        return torch.tensor(0.0, device=feat_map.device, requires_grad=True)
    if pos_idx.numel() > max_samples:
        perm_idx = torch.randperm(pos_idx.numel(), device=feat_map.device)[:max_samples]
        pos_idx = pos_idx[perm_idx]
    feat_edge = feat_flat[pos_idx]
    N = feat_edge.size(0)
    perm = torch.randperm(N, device=feat_edge.device)
    feat_q = feat_edge
    feat_k = feat_edge[perm]
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

    valid = (mask_pool.squeeze(1) > 0.5)  # [B, D', H', W']
    feat_patches = feat_pool.permute(0, 2, 3, 4, 1)[valid].reshape(-1, feat_map.shape[1])
    if feat_patches.size(0) < 2:
        return torch.tensor(0.0, device=feat_map.device, requires_grad=True)
    if uncertainty is not None:
        uncertainty_patches = pool(uncertainty.float()).squeeze(1)[valid].reshape(-1)
        k = min(max_patches, feat_patches.size(0))
        idx = torch.topk(-uncertainty_patches, k).indices  # 低uncertainty优先
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