import torch
import torch.nn.functional as F

def nt_xent_loss(feat_q, feat_k, temperature=0.2):
    """
    feat_q: [N, C] (anchor)
    feat_k: [N, C] (positive)
    """
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