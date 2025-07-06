import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

from dataloaders.la_version1_3 import LAHeart, RandomRotFlip, ElasticDeformation, RandomCrop
from region_classifier import get_edge_and_core_mask, split_to_patches, get_patch_label, UNet3D_Region

# =============== 标注数据增强（大patch增强） ===============
def labeled_aug(image, label, crop_size):
    sample = {'image': image, 'label': label, 'is_labeled': True}
    sample = RandomRotFlip(p=0.3)(sample)
    sample = RandomCrop(crop_size)(sample)
    return sample['image'], sample['label']

# =============== 未标注数据增强（大patch增强） ===============
def unlabeled_aug(image, crop_size):
    sample = {'image': image, 'label': np.zeros_like(image), 'is_labeled': False}
    sample = RandomRotFlip(p=0.8)(sample)
    sample = ElasticDeformation(alpha=15)(sample)
    sample = RandomCrop(crop_size)(sample)
    return sample['image']

# =============== 标注数据集 ===============
class LabeledPatchRegionDataset(Dataset):
    def __init__(
        self,
        laheart_root,
        split='train',
        crop_size=(112, 112, 80),  # 大patch
        patch_size=(16, 16, 16),   # 小patch
        edge_kernel=3,
        edge_thresh=0.15,
        core_thresh=0.5,
        max_num=None,
        patches_per_volume=32
    ):
        self.samples = []
        dataset = LAHeart(base_dir=laheart_root, split=split, num=max_num)
        self.crop_size = crop_size
        self.patch_size = patch_size
        self.edge_kernel = edge_kernel
        self.edge_thresh = edge_thresh
        self.core_thresh = core_thresh
        self.patches_per_volume = patches_per_volume
        for i in tqdm(range(len(dataset)), desc=f"Preparing labeled {split} dataset"):
            sample = dataset[i]
            image, label = sample['image'], sample['label']
            aug_image, aug_label = labeled_aug(image, label, crop_size)
            edge_mask, core_mask = get_edge_and_core_mask(aug_label, kernel_size=edge_kernel)
            # --- split 大patch为小patch ---
            img_patches = split_to_patches(aug_image, patch_size)
            edge_patches = split_to_patches(edge_mask, patch_size)
            core_patches = split_to_patches(core_mask, patch_size)
            perm = np.random.permutation(len(img_patches))
            for idx in perm[:min(self.patches_per_volume, len(img_patches))]:
                ip, ep, cp = img_patches[idx], edge_patches[idx], core_patches[idx]
                plabel = get_patch_label(ep, cp, edge_thresh=edge_thresh, core_thresh=core_thresh)
                if plabel != -1:
                    self.samples.append((ip.astype(np.float32), plabel))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        patch, label = self.samples[idx]
        patch = torch.from_numpy(patch[None, ...])  # (1, D, H, W)
        return patch, torch.tensor(label, dtype=torch.float32)

# =============== 未标注数据集 ===============
class UnlabeledPatchDataset(Dataset):
    def __init__(
        self,
        laheart_root,
        split='train',
        crop_size=(112, 112, 80),  # 大patch
        patch_size=(16, 16, 16),   # 小patch
        max_num=None,
        patches_per_volume=32
    ):
        self.samples = []
        dataset = LAHeart(base_dir=laheart_root, split=split, num=max_num)
        self.crop_size = crop_size
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        for i in tqdm(range(len(dataset)), desc=f"Preparing unlabeled {split} dataset"):
            sample = dataset[i]
            image = sample['image']
            aug_image = unlabeled_aug(image, crop_size)
            img_patches = split_to_patches(aug_image, patch_size)
            perm = np.random.permutation(len(img_patches))
            for idx in perm[:min(self.patches_per_volume, len(img_patches))]:
                self.samples.append(img_patches[idx].astype(np.float32))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        patch = self.samples[idx]
        patch = torch.from_numpy(patch[None, ...])  # (1, D, H, W)
        return patch

# =============== 区域分类器训练 ===============
def train_region_classifier_with_aug(
    laheart_root,
    crop_size=(112, 112, 80),
    patch_size=(16, 16, 16),
    batch_size=32,
    epochs=10,
    lr=1e-3,
    save_name='region_classifier_with_aug.pth'
):
    dataset = LabeledPatchRegionDataset(
        laheart_root, split='train', crop_size=crop_size, patch_size=patch_size
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    model = UNet3D_Region(in_ch=1, base_ch=16).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(epochs):
        total, correct, total_loss = 0, 0, 0
        for patches, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            patches, labels = patches.cuda(), labels.cuda()
            logits = model(patches)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/total:.4f}, Acc: {correct/total:.4f}')
    torch.save(model.state_dict(), save_name)
    print(f"Model saved to {save_name}")
    return model

# =============== 区域分类器测试 ===============
def test_region_classifier_with_aug(
    laheart_root,
    crop_size=(112, 112, 80),
    patch_size=(16, 16, 16),
    batch_size=32,
    model_path='region_classifier_with_aug.pth'
):
    dataset = LabeledPatchRegionDataset(
        laheart_root, split='test', crop_size=crop_size, patch_size=patch_size
    )
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

    acc = (all_labels == all_preds).mean()
    print(f'Test Accuracy: {acc:.4f}')

    print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
    print("Classification Report:\n", classification_report(all_labels, all_preds, target_names=['core', 'edge']))


if __name__ == '__main__':
    laheart_root = '/home/zlj/workspace/tjy/MeTi-SSL/data/2018LA_Seg_Training Set/'
    train_region_classifier_with_aug(
        laheart_root=laheart_root,
        crop_size=(112, 112, 80),
        patch_size=(16, 16, 16),
        batch_size=32,
        epochs=20,
        lr=1e-3,
        save_name='region_classifier_with_aug.pth'
    )
    test_region_classifier_with_aug(
        laheart_root=laheart_root,
        crop_size=(112, 112, 80),
        patch_size=(16, 16, 16),
        batch_size=32,
        model_path='region_classifier_with_aug.pth'
    )