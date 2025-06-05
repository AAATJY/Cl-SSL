import torch
import torch.nn.functional as F

def get_region_masks(pred_soft, uncertainty, threshold=None, edge_kernel=3):
    """
    pred_soft: [B, C, D, H, W] softmax
    uncertainty: [B, 1, D, H, W]
    threshold: float or None, uncertainty门控
    返回 edge_mask, core_mask
    """
    pred = torch.argmax(pred_soft, dim=1, keepdim=True)  # [B,1,D,H,W]
    # morphological edge
    kernel = torch.ones((1,1,edge_kernel,edge_kernel,edge_kernel), device=pred.device)
    dilated = F.conv3d(pred.float(), kernel, padding=edge_kernel//2) > 0
    eroded = F.conv3d(pred.float(), kernel, padding=edge_kernel//2) == kernel.sum()
    edge_mask = (dilated != eroded).float()
    core_mask = (eroded).float()
    if threshold is not None:
        # 动态门控，uncertainty低为核心，高为忽略
        low_mask = (uncertainty < threshold).float()
        core_mask = core_mask * low_mask
        edge_mask = edge_mask * low_mask
    return edge_mask, core_mask