import torch
import torch.nn.functional as F

def extract_voxel_features_by_uncertainty(segmentation_output, feature_maps, low_percent=0.1, high_percent=0.1):
    B, C, D, H, W = segmentation_output.shape
    C_feat = feature_maps.shape[1]

    probs = F.softmax(segmentation_output, dim=1)
    epsilon = 1e-10
    log_probs = torch.log(probs + epsilon)
    entropy = -torch.sum(probs * log_probs, dim=1)   # [B, D, H, W]
    _, pred_class = torch.max(probs, dim=1)          # [B, D, H, W]

    results = []
    for b in range(B):
        batch_entropy = entropy[b]                   # [D, H, W]
        batch_cl = pred_class[b]                     # [D, H, W]
        feats = feature_maps[b].permute(1,2,3,0)     # [D, H, W, C_feat]

        class_results = {}
        for c in range(C):
            mask = (batch_cl == c)
            num_vox = mask.sum().item()
            # 安全判断: 如果当前batch该类别没有体素
            if num_vox == 0:
                class_results[c] = {
                    'low_uncertainty_coords': [],
                    'high_uncertainty_coords': [],
                    'low_uncertainty_features': torch.zeros((0, C_feat), device=feature_maps.device),
                    'high_uncertainty_features': torch.zeros((0, C_feat), device=feature_maps.device)
                }
                continue
            coords = torch.nonzero(mask, as_tuple=True)
            ent_vals = batch_entropy[mask]
            # 再次健壮性检查
            if ent_vals.numel() == 0 or coords[0].numel() == 0:
                class_results[c] = {
                    'low_uncertainty_coords': [],
                    'high_uncertainty_coords': [],
                    'low_uncertainty_features': torch.zeros((0, C_feat), device=feature_maps.device),
                    'high_uncertainty_features': torch.zeros((0, C_feat), device=feature_maps.device)
                }
                continue
            sorted_idx = torch.argsort(ent_vals)
            n_vox = ent_vals.shape[0]
            n_low = max(1, int(low_percent * n_vox))
            n_high = max(1, int(high_percent * n_vox))
            low_idx = sorted_idx[:n_low]
            high_idx = sorted_idx[-n_high:].flip(0)
            z, y, x = coords
            low_coords = list(zip(z[low_idx].tolist(), y[low_idx].tolist(), x[low_idx].tolist()))
            high_coords = list(zip(z[high_idx].tolist(), y[high_idx].tolist(), x[high_idx].tolist()))
            low_feats = feats[z[low_idx], y[low_idx], x[low_idx]]
            high_feats = feats[z[high_idx], y[high_idx], x[high_idx]]
            class_results[c] = {
                'low_uncertainty_coords': low_coords,
                'high_uncertainty_coords': high_coords,
                'low_uncertainty_features': low_feats,
                'high_uncertainty_features': high_feats
            }
        results.append({'batch_idx': b, 'class_results': class_results})
    return results