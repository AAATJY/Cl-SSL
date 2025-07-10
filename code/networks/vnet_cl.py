from torch import nn
import torch
import torch.nn.functional as F


class RegionAwareContrastiveLearning(nn.Module):
    def __init__(self, feat_dim=128, temp=0.1, patch_size=16, edge_threshold=0.25):
        """
        端到端区域感知对比学习模块
        :param feat_dim: 特征维度
        :param temp: 温度参数
        :param patch_size: 补丁大小
        :param edge_threshold: 边缘判定阈值
        """
        super().__init__()
        self.temp = temp
        self.patch_size = patch_size
        self.edge_threshold = edge_threshold

        # 区域分类器 (轻量级)
        self.region_classifier = nn.Sequential(
            nn.Conv3d(feat_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 共享投影头
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, 64)
        )

        # 损失权重
        self.register_buffer('loss_weights', torch.tensor([1.0, 0.7]))  # [patch, voxel]

    def forward(self, anchor_feats, positive_feats, labels=None):
        """
        :param anchor_feats: 锚点特征 [B, C, D, H, W]
        :param positive_feats: 正样本特征 [B, C, D, H, W]
        :param labels: 标签图 [B, D, H, W] (可选)
        """
        B, C, D, H, W = anchor_feats.shape

        # 1. 区域分类
        region_probs = self.region_classifier(anchor_feats)  # [B, 1, D, H, W]
        region_mask = (region_probs > self.edge_threshold).float()

        # 2. 补丁划分
        anchor_patches = self._split_into_patches(anchor_feats)  # [B, N_p, C, P_d, P_h, P_w]
        positive_patches = self._split_into_patches(positive_feats)
        region_patches = self._split_into_patches(region_mask)

        # 3. 初始化损失
        patch_loss = 0
        voxel_loss = 0
        valid_count = 0

        # 4. 遍历所有补丁
        for b in range(B):
            for p_idx in range(anchor_patches.size(1)):
                anchor_patch = anchor_patches[b, p_idx]  # [C, P_d, P_h, P_w]
                positive_patch = positive_patches[b, p_idx]
                region_patch = region_patches[b, p_idx]  # [1, P_d, P_h, P_w]

                # 计算补丁区域类型 (边缘占比)
                edge_ratio = region_patch.mean()

                if edge_ratio > 0.6:  # 边缘区域 - 体素级对比
                    loss = self._voxel_level_contrast(anchor_patch, positive_patch)
                    voxel_loss += loss
                elif edge_ratio < 0.4:  # 核心区域 - 补丁级对比
                    loss = self._patch_level_contrast(anchor_patch, positive_patch)
                    patch_loss += loss
                else:  # 过渡区域 - 跳过
                    continue

                valid_count += 1

        if valid_count == 0:
            return 0

        # 5. 加权损失
        total_loss = (self.loss_weights[0] * patch_loss + self.loss_weights[1] * voxel_loss) / valid_count
        return total_loss

    def _split_into_patches(self, feats):
        """将特征图分割为补丁"""
        B, C, D, H, W = feats.shape
        P_d, P_h, P_w = self.patch_size, self.patch_size, self.patch_size

        # 确保尺寸可整除
        assert D % P_d == 0 and H % P_h == 0 and W % P_w == 0

        # 展开为补丁
        patches = feats.unfold(2, P_d, P_d).unfold(3, P_h, P_h).unfold(4, P_w, P_w)
        patches = patches.contiguous().view(B, -1, C, P_d, P_h, P_w)
        return patches

    def _patch_level_contrast(self, anchor_patch, positive_patch):
        """补丁级对比学习"""
        # 全局平均池化
        anchor_vec = F.adaptive_avg_pool3d(anchor_patch, 1).flatten()  # [C]
        positive_vec = F.adaptive_avg_pool3d(positive_patch, 1).flatten()

        # 投影
        anchor_proj = self.projector(anchor_vec)  # [64]
        positive_proj = self.projector(positive_vec)

        # 归一化
        anchor_proj = F.normalize(anchor_proj, dim=0)
        positive_proj = F.normalize(positive_proj, dim=0)

        # 计算相似度
        logits = torch.dot(anchor_proj, positive_proj) / self.temp
        return -logits  # 最大化相似度

    def _voxel_level_contrast(self, anchor_patch, positive_patch):
        """体素级对比学习 (优化版)"""
        # 展平体素
        anchor_voxels = anchor_patch.flatten(1).permute(1, 0)  # [N_vox, C]
        positive_voxels = positive_patch.flatten(1).permute(1, 0)

        # 随机采样部分体素以节省计算 (最多100个体素)
        num_voxels = anchor_voxels.size(0)
        sample_size = min(100, num_voxels)
        if num_voxels > sample_size:
            indices = torch.randperm(num_voxels)[:sample_size]
            anchor_voxels = anchor_voxels[indices]
            positive_voxels = positive_voxels[indices]

        # 投影和归一化
        anchor_proj = self.projector(anchor_voxels)
        positive_proj = self.projector(positive_voxels)
        anchor_proj = F.normalize(anchor_proj, dim=1)
        positive_proj = F.normalize(positive_proj, dim=1)

        # 计算相似度矩阵
        sim_matrix = torch.mm(anchor_proj, positive_proj.t()) / self.temp

        # 对角线为正样本
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        return F.cross_entropy(sim_matrix, labels)

class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

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
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages-1:
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
        ops.append(nn.Upsample(scale_factor=stride, mode='trilinear',align_corners=False))
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
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,mc_dropout=False, mc_dropout_rate=0.2):
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
                nn.Dropout3d(p=mc_dropout_rate) for _ in range(4)  # 在4个关键位置添加
            ])
        # ========== 新增对比学习组件 ==========
        # 特征提取器（使用编码器最后一层特征）
        self.contrast_feat_extractor = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(n_filters * 16, 128)  # 假设最终特征维度128
        )

        # 区域感知对比学习模块
        self.contrast_learner = RegionAwareContrastiveLearning(
            feat_dim=128,
            patch_size=16,  # 可根据实际调整
            temp=0.1
        )

        # 用于对比学习的额外投影头
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

    def decoder(self, features):
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
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)
        return out

    def forward(self, input, turnoff_drop=False, enable_dropout=True, return_contrast_feats=True):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False

        # MC Dropout特殊处理
        if self.mc_dropout and enable_dropout:
            self.train()  # 强制保持训练模式

        # 编码器特征
        enc_features = self.encoder(input)

        # 对比学习特征 (使用编码器最后一层特征)
        contrast_feats = self.contrast_feat_extractor(enc_features[-1])

        # 解码器特征
        dec_features = self.decoder(enc_features)

        # 分割输出
        seg_out = self.out_conv(dec_features)

        if turnoff_drop:
            self.has_dropout = has_dropout

        if return_contrast_feats:
            return seg_out, contrast_feats
        else:
            return seg_out