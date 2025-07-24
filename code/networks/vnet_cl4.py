import random

import torch
import torch.nn.functional as F
from torch import nn
"""
class RegionAwareContrastiveLearning(nn.Module):
    def __init__(self, feat_dim=128, temp=0.1, patch_size=16, edge_threshold=0.475, hard_neg_k=8):
        super().__init__()
        self.temp = temp
        self.patch_size = patch_size
        self.edge_threshold = edge_threshold
        self.hard_neg_k = hard_neg_k
        self.register_buffer('patch_counts', torch.zeros(3))  # [core, edge, skip]
        self.region_classifier = nn.Sequential(
            nn.Conv3d(feat_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, 64)
        )
        self.register_buffer('loss_weights', torch.tensor([1.0, 0.7]))  # [patch, voxel]

    def forward(self, anchor_feats, positive_feats, labels=None, prob_maps=None):
        self.patch_counts.zero_()
        B, C, D, H, W = anchor_feats.shape
        region_probs = self.region_classifier(anchor_feats)
        region_mask = (region_probs > self.edge_threshold).float()
        anchor_patches = self._split_into_patches(anchor_feats)
        positive_patches = self._split_into_patches(positive_feats)
        region_patches = self._split_into_patches(region_mask)
        if labels is not None:
            label_patches = self._split_into_patches(labels)
        else:
            label_patches = None
        if prob_maps is not None:
            prob_patches = self._split_into_patches(prob_maps)
        else:
            prob_patches = None

        patch_loss = 0
        voxel_loss = 0
        valid_count_patch = 0
        valid_count_voxel = 0

        for b in range(B):
            edge_ratios = region_patches[b].mean(dim=(2, 3, 4))
            for p_idx in range(anchor_patches.size(1)):
                anchor_patch = anchor_patches[b, p_idx]
                positive_patch = positive_patches[b, p_idx]
                edge_ratio = edge_ratios[p_idx].item()
                if edge_ratio > 0.6:
                    self.patch_counts[1] += 1  # 记录边缘区域
                    if label_patches is not None:
                        label_map = label_patches[b, p_idx]
                    else:
                        label_map = None
                    if prob_patches is not None:
                        prob_map = prob_patches[b, p_idx]
                    else:
                        prob_map = region_patches[b, p_idx]
                        # print("边缘= %d",edge_ratio)
                        voxel_loss += self._rcps_voxel_level_contrast(
                            anchor_patch, positive_patch,
                            label_map=label_map, prob_map=prob_map
                        )
                    valid_count_voxel += 1
                elif edge_ratio < 0.4:
                    # print("核心= %d",edge_ratio)
                    self.patch_counts[0] += 1  # 记录核心区域
                    patch_loss += self._patch_level_contrast_batch(
                        anchor_patch, positive_patch, anchor_patches, positive_patches, b, p_idx
                    )
                    valid_count_patch += 1
                else:
                    # print("跳过")
                    self.patch_counts[2] += 1  # 记录跳过区域
                    continue

        patch_loss = patch_loss / max(1, valid_count_patch)
        voxel_loss = voxel_loss / max(1, valid_count_voxel)
        total_loss = self.loss_weights[0] * patch_loss + self.loss_weights[1] * voxel_loss
        print(self.patch_counts)
        return total_loss

    def _split_into_patches(self, feats):
        B, C, D, H, W = feats.shape
        P_d, P_h, P_w = self.patch_size, self.patch_size, self.patch_size

        pad_d = (P_d - D % P_d) % P_d
        pad_h = (P_h - H % P_h) % P_h
        pad_w = (P_w - W % P_w) % P_w
        if pad_d or pad_h or pad_w:
            feats = F.pad(feats, (0, pad_w, 0, pad_h, 0, pad_d))
            D, H, W = feats.shape[-3:]

        patches = feats.unfold(2, P_d, P_d).unfold(3, P_h, P_h).unfold(4, P_w, P_w)
        patches = patches.contiguous().view(B, -1, C, P_d, P_h, P_w)
        return patches

    def _patch_level_contrast_batch(self, anchor_patch, positive_patch, anchor_patches, positive_patches, b, p_idx):
        B, N, C, P_d, P_h, P_w = anchor_patches.shape
        anchor_vec = F.adaptive_avg_pool3d(anchor_patch.unsqueeze(0), 1).squeeze().flatten(0)
        positive_vec = F.adaptive_avg_pool3d(positive_patch.unsqueeze(0), 1).squeeze().flatten(0)
        negatives = []
        for bb in range(B):
            for pp in range(anchor_patches.size(1)):
                if bb == b and pp == p_idx:
                    continue
                neg_patch = anchor_patches[bb, pp]
                neg_vec = F.adaptive_avg_pool3d(neg_patch.unsqueeze(0), 1).squeeze().flatten(0)
                negatives.append(neg_vec)
        if not negatives:
            return torch.tensor(0.0, device=anchor_patch.device)
        negatives = torch.stack(negatives, dim=0)
        anchor_proj = self.projector(anchor_vec)
        pos_proj = self.projector(positive_vec)
        neg_proj = self.projector(negatives)
        anchor_proj = F.normalize(anchor_proj, dim=0)
        pos_proj = F.normalize(pos_proj, dim=0)
        neg_proj = F.normalize(neg_proj, dim=1)
        logits = torch.cat([torch.dot(anchor_proj, pos_proj).unsqueeze(0), torch.mv(neg_proj, anchor_proj)], dim=0)
        logits = logits / self.temp
        loss = -F.log_softmax(logits, dim=0)[0]
        return loss

    def _rcps_voxel_level_contrast(self, anchor_patch, positive_patch, label_map=None, prob_map=None):
        C, D, H, W = anchor_patch.shape
        N = D * H * W
        anchor_vox = anchor_patch.view(C, -1).t()
        positive_vox = positive_patch.view(C, -1).t()

        if prob_map is not None:
            mask = (prob_map.view(-1) > self.edge_threshold)
        else:
            mask = torch.ones(N, dtype=torch.bool, device=anchor_patch.device)
        anchor_sel = anchor_vox[mask]
        positive_sel = positive_vox[mask]
        N_sel = anchor_sel.size(0)
        if N_sel == 0:
            return torch.tensor(0.0, device=anchor_patch.device)

        if label_map is not None:
            labels = label_map.view(-1)[mask]
            pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
            neg_mask = labels.unsqueeze(0) != labels.unsqueeze(1)
        else:
            pos_mask = torch.eye(N_sel, dtype=torch.bool, device=anchor_sel.device)
            neg_mask = ~pos_mask

        anchor_proj = F.normalize(self.projector(anchor_sel), dim=1)
        positive_proj = F.normalize(self.projector(positive_sel), dim=1)
        sim_matrix = torch.mm(anchor_proj, positive_proj.t()) / self.temp

        # 正样本均值
        pos_sim = sim_matrix * pos_mask  # [N_sel, N_sel]
        pos_count = pos_mask.sum(dim=1)  # [N_sel]
        pos_sim_mean = pos_sim.sum(dim=1) / (pos_count + 1e-8)  # [N_sel]

        # hard negative mining保持不变
        hard_negatives = []
        for i in range(N_sel):
            neg_sim = sim_matrix[i][neg_mask[i]]
            if neg_sim.numel() > self.hard_neg_k:
                topk_neg_sim, _ = torch.topk(neg_sim, self.hard_neg_k)
                hard_negatives.append(topk_neg_sim)
            else:
                hard_negatives.append(neg_sim)
        hard_negatives = torch.stack([F.pad(hn, (0, self.hard_neg_k - hn.numel()), value=0) for hn in hard_negatives],
                                    dim=0)  # [N_sel, hard_neg_k]
        neg_sim = hard_negatives

        exp_pos = torch.exp(pos_sim_mean)
        exp_neg = torch.exp(neg_sim).sum(dim=1) + exp_pos
        loss = -torch.log(exp_pos / (exp_neg + 1e-8))
        return loss.mean()
"""

class RegionAwareContrastiveLearning(nn.Module):
    def __init__(self, feat_dim=128, temp=0.1, patch_size=16, edge_threshold=0.4,
                 hard_neg_k=32, patch_sample_k=12, edge_patch_sample_k=24, voxel_sample_k=256):
        super().__init__()
        self.temp = temp
        self.patch_size = patch_size
        self.edge_threshold = edge_threshold
        self.hard_neg_k = hard_neg_k
        self.patch_sample_k = patch_sample_k
        self.edge_patch_sample_k = edge_patch_sample_k
        self.voxel_sample_k = voxel_sample_k
        self.register_buffer('patch_counts', torch.zeros(3))  # [core, edge, skip]
        self.region_classifier = nn.Sequential(
            nn.Conv3d(feat_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, 64)
        )
        self.register_buffer('loss_weights', torch.tensor([1.0, 0.7]))  # [patch, voxel]

    def forward(self, anchor_feats, positive_feats, labels=None, prob_maps=None):
        self.patch_counts.zero_()
        B, C, D, H, W = anchor_feats.shape
        region_probs = self.region_classifier(anchor_feats)
        region_mask = (region_probs > self.edge_threshold).float()
        anchor_patches = self._split_into_patches(anchor_feats)
        positive_patches = self._split_into_patches(positive_feats)
        region_patches = self._split_into_patches(region_mask)
        if labels is not None:
            label_patches = self._split_into_patches(labels)
        else:
            label_patches = None
        if prob_maps is not None:
            prob_patches = self._split_into_patches(prob_maps)
        else:
            prob_patches = None

        patch_loss = 0
        voxel_loss = 0
        valid_count_patch = 0
        valid_count_voxel = 0

        for b in range(B):
            edge_ratios = region_patches[b].mean(dim=(2, 3, 4))
            core_patch_indices = [p_idx for p_idx, edge_ratio in enumerate(edge_ratios) if edge_ratio < 0.35]
            edge_patch_indices = [p_idx for p_idx, edge_ratio in enumerate(edge_ratios) if edge_ratio > 0.65]
            skip_patch_indices = [p_idx for p_idx, edge_ratio in enumerate(edge_ratios) if 0.35 <= edge_ratio <= 0.65]

            # ---- patch-level采样（核心区域）----
            if len(core_patch_indices) > self.patch_sample_k:
                sampled_core_patch_indices = random.sample(core_patch_indices, self.patch_sample_k)
            else:
                sampled_core_patch_indices = core_patch_indices
            # ---- voxel-level采样（边缘区域）----
            if len(edge_patch_indices) > self.edge_patch_sample_k:
                sampled_edge_patch_indices = random.sample(edge_patch_indices, self.edge_patch_sample_k)
            else:
                sampled_edge_patch_indices = edge_patch_indices

            for p_idx in sampled_core_patch_indices:
                anchor_patch = anchor_patches[b, p_idx]
                positive_patch = positive_patches[b, p_idx]
                patch_loss += self._patch_level_contrast_batch(
                    anchor_patch, positive_patch, anchor_patches, positive_patches, b, p_idx
                )
                valid_count_patch += 1
                self.patch_counts[0] += 1  # 核心区域统计

            for p_idx in sampled_edge_patch_indices:
                anchor_patch = anchor_patches[b, p_idx]
                positive_patch = positive_patches[b, p_idx]
                label_map = label_patches[b, p_idx] if label_patches is not None else None
                prob_map = prob_patches[b, p_idx] if prob_patches is not None else region_patches[b, p_idx]
                voxel_loss += self._rcps_voxel_level_contrast(
                    anchor_patch, positive_patch,
                    label_map=label_map, prob_map=prob_map,
                    voxel_sample_k=self.voxel_sample_k
                )
                valid_count_voxel += 1
                self.patch_counts[1] += 1  # 边缘区域统计

            for p_idx in skip_patch_indices:
                self.patch_counts[2] += 1  # 跳过区域统计

        patch_loss = patch_loss / max(1, valid_count_patch)
        voxel_loss = voxel_loss / max(1, valid_count_voxel)
        total_loss = self.loss_weights[0] * patch_loss + self.loss_weights[1] * voxel_loss
        print(self.patch_counts)
        return total_loss

    def _split_into_patches(self, feats):
        B, C, D, H, W = feats.shape
        P_d, P_h, P_w = self.patch_size, self.patch_size, self.patch_size

        pad_d = (P_d - D % P_d) % P_d
        pad_h = (P_h - H % P_h) % P_h
        pad_w = (P_w - W % P_w) % P_w
        if pad_d or pad_h or pad_w:
            feats = F.pad(feats, (0, pad_w, 0, pad_h, 0, pad_d))
            D, H, W = feats.shape[-3:]

        patches = feats.unfold(2, P_d, P_d).unfold(3, P_h, P_h).unfold(4, P_w, P_w)
        patches = patches.contiguous().view(B, -1, C, P_d, P_h, P_w)
        return patches

    def _patch_level_contrast_batch(self, anchor_patch, positive_patch, anchor_patches, positive_patches, b, p_idx):
        B, N, C, P_d, P_h, P_w = anchor_patches.shape
        anchor_vec = F.adaptive_avg_pool3d(anchor_patch.unsqueeze(0), 1).squeeze().flatten(0)
        positive_vec = F.adaptive_avg_pool3d(positive_patch.unsqueeze(0), 1).squeeze().flatten(0)
        negatives = []
        negative_indices = [(bb, pp) for bb in range(B) for pp in range(anchor_patches.size(1)) if not (bb == b and pp == p_idx)]
        if len(negative_indices) > self.patch_sample_k:
            negative_indices = random.sample(negative_indices, self.patch_sample_k)
        for bb, pp in negative_indices:
            neg_patch = anchor_patches[bb, pp]
            neg_vec = F.adaptive_avg_pool3d(neg_patch.unsqueeze(0), 1).squeeze().flatten(0)
            negatives.append(neg_vec)
        if not negatives:
            return torch.tensor(0.0, device=anchor_patch.device)
        negatives = torch.stack(negatives, dim=0)
        anchor_proj = self.projector(anchor_vec)
        pos_proj = self.projector(positive_vec)
        neg_proj = self.projector(negatives)
        anchor_proj = F.normalize(anchor_proj, dim=0)
        pos_proj = F.normalize(pos_proj, dim=0)
        neg_proj = F.normalize(neg_proj, dim=1)
        logits = torch.cat([torch.dot(anchor_proj, pos_proj).unsqueeze(0), torch.mv(neg_proj, anchor_proj)], dim=0)
        logits = logits / self.temp
        loss = -F.log_softmax(logits, dim=0)[0]
        return loss

    def _rcps_voxel_level_contrast(self, anchor_patch, positive_patch, label_map=None, prob_map=None, voxel_sample_k=256):
        C, D, H, W = anchor_patch.shape
        N = D * H * W
        anchor_vox = anchor_patch.view(C, -1).t()
        positive_vox = positive_patch.view(C, -1).t()

        if prob_map is not None:
            mask = (prob_map.view(-1) > self.edge_threshold)
        else:
            mask = torch.ones(N, dtype=torch.bool, device=anchor_patch.device)
        anchor_sel = anchor_vox[mask]
        positive_sel = positive_vox[mask]
        N_sel = anchor_sel.size(0)
        if N_sel == 0:
            return torch.tensor(0.0, device=anchor_patch.device)

        # ---- 随机采样 voxel ----
        if N_sel > voxel_sample_k:
            sample_idx = torch.randperm(N_sel, device=anchor_sel.device)[:voxel_sample_k]
            anchor_sel = anchor_sel[sample_idx]
            positive_sel = positive_sel[sample_idx]
            N_sel = voxel_sample_k
            if label_map is not None:
                labels = label_map.view(-1)[mask][sample_idx]
            else:
                labels = None
        else:
            if label_map is not None:
                labels = label_map.view(-1)[mask]
            else:
                labels = None

        if labels is not None:
            pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
            neg_mask = labels.unsqueeze(0) != labels.unsqueeze(1)
        else:
            pos_mask = torch.eye(N_sel, dtype=torch.bool, device=anchor_sel.device)
            neg_mask = ~pos_mask

        anchor_proj = F.normalize(self.projector(anchor_sel), dim=1)
        positive_proj = F.normalize(self.projector(positive_sel), dim=1)
        sim_matrix = torch.mm(anchor_proj, positive_proj.t()) / self.temp

        # 正样本均值
        pos_sim = sim_matrix * pos_mask  # [N_sel, N_sel]
        pos_count = pos_mask.sum(dim=1)  # [N_sel]
        pos_sim_mean = pos_sim.sum(dim=1) / (pos_count + 1e-8)  # [N_sel]

        # hard negative mining
        hard_negatives = []
        for i in range(N_sel):
            neg_sim = sim_matrix[i][neg_mask[i]]
            if neg_sim.numel() > self.hard_neg_k:
                topk_neg_sim, _ = torch.topk(neg_sim, self.hard_neg_k)
                hard_negatives.append(topk_neg_sim)
            else:
                hard_negatives.append(neg_sim)
        hard_negatives = torch.stack([F.pad(hn, (0, self.hard_neg_k - hn.numel()), value=0) for hn in hard_negatives],
                                    dim=0)  # [N_sel, hard_neg_k]
        neg_sim = hard_negatives

        exp_pos = torch.exp(pos_sim_mean)
        exp_neg = torch.exp(neg_sim).sum(dim=1) + exp_pos
        loss = -torch.log(exp_pos / (exp_neg + 1e-8))
        return loss.mean()
# 其余Block结构保持不变...

class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()
        ops = []
        for i in range(n_stages):
            input_channel = n_filters_in if i == 0 else n_filters_out
            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)
    def forward(self, x):
        x = self.conv(x)
        return x

class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()
        ops = []
        for i in range(n_stages):
            input_channel = n_filters_in if i == 0 else n_filters_out
            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            if i != n_stages - 1:
                ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x

class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()
        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)
    def forward(self, x):
        x = self.conv(x)
        return x

class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()
        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)
    def forward(self, x):
        x = self.conv(x)
        return x

class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(Upsampling, self).__init__()
        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode='trilinear', align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)
    def forward(self, x):
        x = self.conv(x)
        return x

class VNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 mc_dropout=False, mc_dropout_rate=0.2):
        super(VNet, self).__init__()
        self.has_dropout = has_dropout
        self.mc_dropout = mc_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        if has_dropout:
            self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        if mc_dropout:  # 教师模型专用dropout
            self.mc_dropout_layers = nn.ModuleList([
                nn.Dropout3d(p=mc_dropout_rate) for _ in range(4)
            ])
        # ========== 新增对比学习组件 ==========
        self.decoder_proj = nn.Conv3d(n_filters, 128, 1)  # 新增：将解码器输出投影到256维
        self.contrast_feat_extractor = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(n_filters*16, 128)
        )
        self.contrast_learner = RegionAwareContrastiveLearning(
            feat_dim=128,
            patch_size=16,
            temp=0.1
        )
        self.contrast_projector = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)
        if self.mc_dropout:
            x1 = self.mc_dropout_layers[0](x1)
        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)
        if self.mc_dropout:
            x2 = self.mc_dropout_layers[1](x2)
        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)
        if self.mc_dropout:
            x3 = self.mc_dropout_layers[2](x3)
        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)
        if self.mc_dropout:
            x4 = self.mc_dropout_layers[3](x4)
        x5 = self.block_five(x4_dw)
        if self.has_dropout:
            x5 = self.dropout(x5)
        res = [x1, x2, x3, x4, x5]
        return res

    def decoder(self, features, return_spatial_feats=False):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1

        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)
        if return_spatial_feats:
            return x9  # [B, n_filters, D, H, W]
        return x9

    def forward(self, input, turnoff_drop=False, enable_dropout=True, return_contrast_feats=True,
                return_encoder_feats=False, return_decoder_feats=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        if self.mc_dropout and enable_dropout:
            self.train()
        enc_features = self.encoder(input)
        contrast_feats = self.contrast_feat_extractor(enc_features[-1])
        if return_decoder_feats:
            dec_features = self.decoder(enc_features, return_spatial_feats=True)
            dec_features_proj = self.decoder_proj(dec_features)
        else:
            dec_features = self.decoder(enc_features, return_spatial_feats=False)
            dec_features_proj = None
        seg_out = self.out_conv(dec_features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        if return_decoder_feats:
            # 返回：分割输出，全局对比特征，解码器空间特征（投影后，通道256）
            return seg_out, contrast_feats, dec_features_proj
        elif return_encoder_feats:
            return seg_out, contrast_feats, enc_features[-1]
        elif return_contrast_feats:
            return seg_out, contrast_feats
        else:
            return seg_out