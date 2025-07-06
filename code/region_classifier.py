import os
import h5py
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

from dataloaders.la_version1_3 import LAHeart
from scipy.ndimage import binary_erosion
from sklearn.metrics import confusion_matrix, classification_report

# ----------------- 数据加载，参考LAHeart -----------------
class LAHeart(Dataset):
    def __init__(self, base_dir=None, split='train', num=None):
        self._base_dir = base_dir
        if split == 'train':
            with open(os.path.join(self._base_dir, '../train.list'), 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(os.path.join(self._base_dir, '../test.list'), 'r') as f:
                self.image_list = f.readlines()
        self.image_list = [item.strip() for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("Total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(os.path.join(self._base_dir, image_name, 'mri_norm2.h5'), 'r')
        image = np.ascontiguousarray(h5f['image'][:])  # shape (D, H, W)
        label = np.ascontiguousarray(h5f['label'][:])
        return image, label

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
    D, H, W = img.shape
    patches = []
    for z in range(0, D - patch_size + 1, patch_size):
        for y in range(0, H - patch_size + 1, patch_size):
            for x in range(0, W - patch_size + 1, patch_size):
                patch = img[z:z+patch_size, y:y+patch_size, x:x+patch_size]
                patches.append(patch)
    return np.stack(patches, axis=0) if len(patches) > 0 else np.empty((0, patch_size, patch_size, patch_size))

def get_patch_label(edge_patch, core_patch, edge_thresh=0.2, core_thresh=0.5):
    edge_ratio = edge_patch.sum() / edge_patch.size
    core_ratio = core_patch.sum() / core_patch.size
    if edge_ratio > edge_thresh:
        return 1  # 边缘
    if core_ratio > core_thresh:
        return 0  # 核心
    return -1

# ----------------- Patch区域二分类数据集 -----------------
class PatchRegionDataset(Dataset):
    def __init__(self, laheart_root, split='train', patch_size=16, edge_kernel=3, edge_thresh=0.14, core_thresh=0.6, max_num=None):
        self.samples = []
        dataset = LAHeart(base_dir=laheart_root, split=split, num=max_num)
        for i in tqdm(range(len(dataset)), desc="Preparing region dataset"):
            image, label = dataset[i]
            edge_mask, core_mask = get_edge_and_core_mask(label, kernel_size=edge_kernel)
            img_patches = split_to_patches(image, patch_size)
            edge_patches = split_to_patches(edge_mask, patch_size)
            core_patches = split_to_patches(core_mask, patch_size)
            for ip, ep, cp in zip(img_patches, edge_patches, core_patches):
                plabel = get_patch_label(ep, cp, edge_thresh=edge_thresh, core_thresh=core_thresh)
                if plabel != -1:
                    self.samples.append((ip.astype(np.float32), plabel))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        patch, label = self.samples[idx]
        patch = torch.from_numpy(patch[None, ...])  # (1, D, H, W)
        return patch, torch.tensor(label, dtype=torch.float32)

# ----------------- 3D UNet骨干做二分类 -----------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class UNet3D_Region(nn.Module):
    def __init__(self, in_ch=1, base_ch=16):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base_ch)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = DoubleConv(base_ch, base_ch*2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = DoubleConv(base_ch*2, base_ch*4)
        self.pool3 = nn.MaxPool3d(2)
        self.bottleneck = DoubleConv(base_ch*4, base_ch*8)
        # 分类头
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(base_ch*8, 1)
    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.bottleneck(self.pool3(x3))
        feat = self.avgpool(x4).flatten(1)
        out = self.fc(feat).squeeze(-1)
        return out

# ----------------- 训练主流程 -----------------
def train_region_classifier(laheart_root, patch_size=16, batch_size=32, epochs=10, lr=1e-3,
                            save_name='region_classifier_unet.pth', boundary_weight=0.85):
    dataset = PatchRegionDataset(laheart_root, split='train', patch_size=patch_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    model = UNet3D_Region(in_ch=1, base_ch=16).cuda()

    # 统计类别数量
    labels = [label for _, label in dataset.samples]
    n_core = sum(np.array(labels) == 0)
    n_edge = sum(np.array(labels) == 1)
    pos_weight = torch.tensor([n_core / (n_edge + 1e-6)]).cuda()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total, correct, total_loss = 0, 0, 0
        for patches, labels in tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            patches, labels = patches.cuda(), labels.cuda()
            logits = model(patches)
            # 主损失
            loss_main = loss_fn(logits, labels)
            # 边界损失（只对label==1样本）
            edge_mask = (labels == 1)
            if edge_mask.any():
                edge_logits = logits[edge_mask]
                edge_labels = labels[edge_mask]
                loss_edge = loss_fn(edge_logits, edge_labels)
                loss = loss_main + boundary_weight * loss_edge
            else:
                loss = loss_main

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / total:.4f}, Acc: {correct / total:.4f}')
    torch.save(model.state_dict(), save_name)
    print(f"Model saved to {save_name}")
    return model
# ----------------- 测试主流程 -----------------
class PatchRegionDatasetTest(Dataset):
    def __init__(self, laheart_root, split='test', patch_size=16, edge_kernel=3, edge_thresh=0.14, core_thresh=0.6, max_num=None):
        self.samples = []
        dataset = LAHeart(base_dir=laheart_root, split=split, num=max_num)
        for i in tqdm(range(len(dataset)), desc="Preparing region test dataset"):
            image, label = dataset[i]
            edge_mask, core_mask = get_edge_and_core_mask(label, kernel_size=edge_kernel)
            img_patches = split_to_patches(image, patch_size)
            edge_patches = split_to_patches(edge_mask, patch_size)
            core_patches = split_to_patches(core_mask, patch_size)
            for ip, ep, cp in zip(img_patches, edge_patches, core_patches):
                plabel = get_patch_label(ep, cp, edge_thresh=edge_thresh, core_thresh=core_thresh)
                if plabel != -1:
                    self.samples.append((ip.astype(np.float32), plabel))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        patch, label = self.samples[idx]
        patch = torch.from_numpy(patch[None, ...])  # (1, D, H, W)
        return patch, torch.tensor(label, dtype=torch.float32)

def test_region_classifier(laheart_root, model_path, patch_size=16, batch_size=32):
    dataset = PatchRegionDatasetTest(laheart_root, split='test', patch_size=patch_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet3D_Region(in_ch=1, base_ch=16).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for patches, labels in tqdm(loader, desc="Testing"):
            patches = patches.to(device)
            logits = model(patches)
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    all_labels = np.concatenate(all_labels).astype(int)
    all_preds = np.concatenate(all_preds).astype(int)

    # 总体准确率
    acc = (all_labels == all_preds).mean()
    print(f'Test Accuracy: {acc:.4f}')

    # 混淆矩阵和每类准确率
    print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
    print("Classification Report:\n", classification_report(all_labels, all_preds, target_names=['core', 'edge']))

# --------- 用法示例 ----------
if __name__ == '__main__':
    laheart_root = '/home/zlj/workspace/tjy/MeTi-SSL/data/2018LA_Seg_Training Set/'
    train_region_classifier(laheart_root, patch_size=16, batch_size=32, epochs=35, lr=1e-3)
    model_path = 'region_classifier_unet.pth'  # 训练保存的模型路径
    test_region_classifier(laheart_root, model_path, patch_size=16, batch_size=32)