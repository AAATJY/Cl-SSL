import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()
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
            ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


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
        return self.conv(x)


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
        return self.conv(x)


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
        return self.conv(x)


class VNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none',
                 has_dropout=False, mc_dropout=False, mc_dropout_rate=0.2):
        super(VNet, self).__init__()
        self.has_dropout = has_dropout
        self.mc_dropout = mc_dropout

        # 编码器部分
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

        # 解码器部分
        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        # Dropout 配置
        if has_dropout:
            self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        if mc_dropout:
            self.mc_dropout_layers = nn.ModuleList([
                nn.Dropout3d(p=mc_dropout_rate) for _ in range(4)
            ])

    def encoder(self, input):
        """编码器部分，返回多尺度特征"""
        x1 = self.block_one(input)  # 低层特征
        x1_dw = self.block_one_dw(x1)
        if self.mc_dropout:
            x1 = self.mc_dropout_layers[0](x1)

        x2 = self.block_two(x1_dw)  # 中低层特征
        x2_dw = self.block_two_dw(x2)
        if self.mc_dropout:
            x2 = self.mc_dropout_layers[1](x2)

        x3 = self.block_three(x2_dw)  # 中层特征
        x3_dw = self.block_three_dw(x3)
        if self.mc_dropout:
            x3 = self.mc_dropout_layers[2](x3)

        x4 = self.block_four(x3_dw)  # 中高层特征
        x4_dw = self.block_four_dw(x4)
        if self.mc_dropout:
            x4 = self.mc_dropout_layers[3](x4)

        x5 = self.block_five(x4_dw)  # 高层特征
        if self.has_dropout:
            x5 = self.dropout(x5)

        # 返回所有尺度的特征
        return {
            'low_level': x1,  # 低层特征 (n_filters, 112, 112, 80)
            'mid_low_level': x2,  # 中低层特征 (2*n_filters, 56, 56, 40)
            'mid_level': x3,  # 中层特征 (4*n_filters, 28, 28, 20)
            'mid_high_level': x4,  # 中高层特征 (8*n_filters, 14, 14, 10)
            'high_level': x5  # 高层特征 (16*n_filters, 7, 7, 5)
        }

    def decoder(self, features):
        """解码器部分，使用多尺度特征"""
        x1 = features['low_level']
        x2 = features['mid_low_level']
        x3 = features['mid_level']
        x4 = features['mid_high_level']
        x5 = features['high_level']

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

        x9 = self.block_nine(x8_up)  # 最终特征

        if self.has_dropout:
            x9 = self.dropout(x9)

        out = self.out_conv(x9)
        return out, x9

    def forward(self, input, turnoff_drop=False, enable_dropout=True,
                return_features=False, return_multi_scale=False):
        """
        前向传播

        Args:
            input: 输入张量
            turnoff_drop: 是否关闭dropout
            enable_dropout: 是否启用MC dropout
            return_features: 是否返回最后一层特征
            return_multi_scale: 是否返回多尺度特征

        Returns:
            根据参数返回不同的输出
        """
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False

        if self.mc_dropout and enable_dropout:
            self.train()

        # 编码器提取多尺度特征
        multi_scale_features = self.encoder(input)

        # 解码器生成输出
        out, final_feature = self.decoder(multi_scale_features)

        if turnoff_drop:
            self.has_dropout = has_dropout

        # 根据参数返回不同的输出
        if return_multi_scale:
            # 返回所有尺度的特征（用于AMR-CMB）
            return out, multi_scale_features
        elif return_features:
            # 返回输出和最终特征（用于原始CFCMB）
            return out, final_feature
        else:
            # 只返回输出
            return out