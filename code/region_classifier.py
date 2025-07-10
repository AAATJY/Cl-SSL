import os
import torch.nn as nn
import numpy as np

# ----------------- 形态学生成边缘和核心 -----------------
from scipy.ndimage import binary_erosion

def get_edge_and_core_mask(label, kernel_size=3):
    fg_mask = (label > 0).astype(np.uint8)
    structure = np.ones((kernel_size, kernel_size, kernel_size))
    eroded = binary_erosion(fg_mask, structure=structure)
    edge_mask = fg_mask ^ eroded
    core_mask = eroded.astype(np.uint8)
    return edge_mask, core_mask

# ----------------- Patch切分和标签 -----------------
def split_to_patches(img, patch_size):
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size, patch_size)
    D, H, W = img.shape
    pD, pH, pW = patch_size
    patches = []
    for z in range(0, D - pD + 1, pD):
        for y in range(0, H - pH + 1, pH):
            for x in range(0, W - pW + 1, pW):
                patch = img[z:z+pD, y:y+pH, x:x+pW]
                patches.append(patch)
    if len(patches) > 0:
        return np.stack(patches, axis=0)
    else:
        return np.empty((0, pD, pH, pW))

def get_patch_label(edge_patch, core_patch, edge_thresh=0.2, core_thresh=0.5):
    edge_ratio = edge_patch.sum() / edge_patch.size
    core_ratio = core_patch.sum() / core_patch.size
    if edge_ratio > edge_thresh:
        return 1  # 边缘
    if core_ratio > core_thresh:
        return 0  # 核心
    return -1


# ----------------- 3D UNet骨干做二分类 -----------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, norm='bn'):
        super().__init__()
        norm_layer = nn.BatchNorm3d if norm == 'bn' else nn.LayerNorm
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            norm_layer(out_ch) if norm == 'bn' else norm_layer([out_ch, 1, 1, 1]),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            norm_layer(out_ch) if norm == 'bn' else norm_layer([out_ch, 1, 1, 1]),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class DownConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        return self.down(x)

class UNet3D_Region(nn.Module):
    def __init__(self, in_ch=1, base_ch=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base_ch)
        self.down1 = DownConv(base_ch, base_ch * 2)
        self.enc2 = DoubleConv(base_ch * 2, base_ch * 2)
        self.down2 = DownConv(base_ch * 2, base_ch * 4)
        self.enc3 = DoubleConv(base_ch * 4, base_ch * 4)
        self.down3 = DownConv(base_ch * 4, base_ch * 8)
        self.enc4 = DoubleConv(base_ch * 8, base_ch * 8)

        # 分类头
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(base_ch * 8, 1)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(self.down1(x))
        x = self.enc3(self.down2(x))
        x = self.enc4(self.down3(x))
        feat = self.avgpool(x).flatten(1)
        feat = self.dropout(feat)
        out = self.fc(feat).squeeze(-1)
        return out
# class DoubleConv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv3d(in_ch, out_ch, 3, padding=1),
#             nn.BatchNorm3d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(out_ch, out_ch, 3, padding=1),
#             nn.BatchNorm3d(out_ch),
#             nn.ReLU(inplace=True)
#         )
#     def forward(self, x):
#         return self.net(x)
#
# class UNet3D_Region(nn.Module):
#     def __init__(self, in_ch=1, base_ch=16):
#         super().__init__()
#         self.enc1 = DoubleConv(in_ch, base_ch)
#         self.pool1 = nn.MaxPool3d(2)
#         self.enc2 = DoubleConv(base_ch, base_ch*2)
#         self.pool2 = nn.MaxPool3d(2)
#         self.enc3 = DoubleConv(base_ch*2, base_ch*4)
#         self.pool3 = nn.MaxPool3d(2)
#         self.bottleneck = DoubleConv(base_ch*4, base_ch*8)
#         # 分类头
#         self.avgpool = nn.AdaptiveAvgPool3d(1)
#         self.fc = nn.Linear(base_ch*8, 1)
#     def forward(self, x):
#         x1 = self.enc1(x)
#         x2 = self.enc2(self.pool1(x1))
#         x3 = self.enc3(self.pool2(x2))
#         x4 = self.bottleneck(self.pool3(x3))
#         feat = self.avgpool(x4).flatten(1)
#         out = self.fc(feat).squeeze(-1)
#         return out