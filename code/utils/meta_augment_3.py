import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from dataloaders.la_version1_3 import (
    RandomRotFlip, RandomCrop, ElasticDeformation, GaussianBlur, ContrastAdjust,
    GammaCorrection, LocalShuffle, RandomNoise, RandomOcclusion, EdgeEnhancement,
    MotionArtifact, CutMix3D, MixUp3D, ToTensor
)


class MetaAugController(nn.Module):
    def __init__(self, num_aug, init_temp=0.1, init_weights=None):
        super().__init__()
        if init_weights is not None:
            if len(init_weights) != num_aug:
                raise ValueError(f"init_weights长度({len(init_weights)})必须与num_aug({num_aug})一致")
            self.weights = nn.Parameter(torch.tensor(init_weights, dtype=torch.float32))
        else:
            self.weights = nn.Parameter(torch.ones(num_aug))
        self.temperature = init_temp
        self.optimizer = optim.Adam([self.weights], lr=1e-4)
        self.history = []

    def get_probs(self):
        return F.softmax(self.weights / self.temperature, dim=-1)

    def record_batch(self, aug_indices):
        self.history.append({
            'indices': aug_indices,
        })

    def update_weights(self, sample_loss):
        self.optimizer.zero_grad()
        grad_dict = {i: 0.0 for i in range(len(self.weights))}
        probs = self.get_probs()
        for loss_value, record in zip(sample_loss, self.history):
            aug_indices = record['indices']
            loss_value = loss_value.unsqueeze(0)
            weighted_loss = loss_value * probs[aug_indices].sum()
            grads = torch.autograd.grad(
                outputs=weighted_loss,
                inputs=self.weights,
                retain_graph=True,
                create_graph=False,
                allow_unused=True
            )[0]
            if grads is None:
                continue
            for i in range(len(grads)):
                if i not in aug_indices:
                    grads[i] = 0
            for idx in aug_indices:
                grad_dict[idx] += grads[idx].item()

        if any(g != 0 for g in grad_dict.values()):
            grad_tensor = torch.tensor(
                [grad_dict[i] for i in range(len(self.weights))],
                device=self.weights.device
            )
            self.weights.grad = grad_tensor
            torch.nn.utils.clip_grad_norm_([self.weights], 5.0)
            self.optimizer.step()
        with torch.no_grad():
            s = self.weights.data.sum()
            if s.abs() > 1e-8:
                self.weights.data = self.weights.data / s
        self.history = []


class AugmentationFactory:
    @staticmethod
    def weak_base_aug(patch_size):
        return transforms.Compose([
            RandomRotFlip(p=0.3),
            RandomCrop(patch_size)
        ])

    @staticmethod
    def strong_base_aug(patch_size):
        return transforms.Compose([
            RandomRotFlip(p=0.8),
            ElasticDeformation(alpha=15),
            RandomCrop(patch_size)
        ])

    @staticmethod
    def get_weak_weighted_augs():
        # 共5个
        return [
            GaussianBlur(sigma_range=(0.5, 1.0)),
            ContrastAdjust(factor_range=(0.8, 1.2)),
            GammaCorrection(gamma_range=(0.8, 1.2)),
            LocalShuffle(max_ratio=0.05, block_size=8),
            RandomNoise(sigma=0.05)
        ]

    @staticmethod
    def get_strong_weighted_augs():
        # 共6个
        return [
            GaussianBlur(sigma_range=(1.0, 2.0)),
            GammaCorrection(gamma_range=(0.5, 2.5)),
            LocalShuffle(max_ratio=0.2, block_size=16),
            EdgeEnhancement(),
            RandomNoise(sigma=0.2),
            ContrastAdjust(factor_range=(0.8, 1.2)),
        ]

    @staticmethod
    def get_intensity_only_augs():
        # 共5个（仅强度增强，保证几何对齐）
        return [
            GaussianBlur(sigma_range=(0.8, 1.6)),
            GammaCorrection(gamma_range=(0.7, 1.6)),
            ContrastAdjust(factor_range=(0.8, 1.2)),
            RandomNoise(sigma=0.1),
            EdgeEnhancement(),
        ]


class WeightedWeakAugment(nn.Module):
    def __init__(self, aug_list, controller=None, alpha=1.0):
        super().__init__()
        self.augmenters = aug_list
        self.controller = controller
        self.alpha = alpha
        # 默认权重为5个元素；若与增强器数量不同，get_aug_weights 内部会统一处理
        self.default_weights = torch.tensor([0.15, 0.15, 0.2, 0.2, 0.3], dtype=torch.float32)

    def get_aug_weights(self):
        n = len(self.augmenters)
        # 没有控制器：用默认权重；若长度不匹配，回退为均匀分布
        if self.controller is None:
            w = self.default_weights.detach().cpu().numpy()
            if w.size != n or np.sum(w) <= 0:
                return np.ones(n, dtype=np.float32) / max(n, 1)
            return (w / np.sum(w)).astype(np.float32)

        # 有控制器：优先用 controller.get_probs()，否则直接用 weights
        try:
            if hasattr(self.controller, 'get_probs'):
                w_t = self.controller.get_probs().detach().cpu()
            else:
                w_t = self.controller.weights.detach().cpu()
            w = w_t.numpy().astype(np.float32)
        except Exception:
            w = np.ones(n, dtype=np.float32)

        # 对齐长度：过长裁剪，过短填充，再归一化
        if w.size != n:
            if w.size > n:
                w = w[:n]
            else:
                # 用均匀值填充，避免偏置
                pad = np.ones(n - w.size, dtype=np.float32)
                w = np.concatenate([w, pad], axis=0)
        w = np.maximum(w, 1e-8)
        w = w / np.sum(w)
        return w

    def forward(self, sample, sample_pair=None):
        original = sample.copy()
        weights = self.get_aug_weights()
        aug_idx = np.random.choice(len(self.augmenters), p=weights)
        augmenter = self.augmenters[aug_idx]

        if isinstance(augmenter, (CutMix3D, MixUp3D)):
            if sample_pair is not None:
                aug_sample = augmenter(original.copy(), sample_pair)
            else:
                aug_sample = original
        else:
            aug_sample = augmenter(original.copy())
            mix_ratio = np.random.beta(0.3 + self.alpha, 0.3 + (1 - self.alpha))
            aug_sample['image'] = (1 - mix_ratio) * original['image'] + mix_ratio * aug_sample['image']

        if 'label' in original:
            aug_sample['label'] = original['label']

        if self.controller is not None:
            aug_sample['aug_idx'] = aug_idx
        return aug_sample


class DualTransformWrapper:
    def __init__(self, labeled_aug, unlabeled_aug, controller=None, paired_sample=None):
        self.labeled_aug = labeled_aug
        self.unlabeled_aug = unlabeled_aug
        self.controller = controller

    def __call__(self, sample, paired_sample=None):
        if sample.get('is_labeled', True):
            return self.labeled_aug(sample)
        else:
            if callable(self.unlabeled_aug) and 'sample_pair' in self.unlabeled_aug.__call__.__code__.co_varnames:
                return self.unlabeled_aug(sample, sample_pair=paired_sample)
            else:
                return self.unlabeled_aug(sample)


def random_pair_indices(length):
    indices = np.arange(length)
    pair_indices = []
    for i in range(length):
        choices = list(indices)
        choices.remove(i)
        pair_indices.append(np.random.choice(choices))
    return pair_indices


def batch_aug_wrapper(batch_data, labeled_aug_in, unlabeled_aug_in, controller=None):
    images = batch_data['image'].numpy()
    labels = batch_data['label'].numpy()
    is_labeled = batch_data.get('is_labeled', [True] * len(images))
    batch_size = len(images)
    sampled_batch = []
    aug_indices = []

    pair_indices = random_pair_indices(batch_size)

    for idx, (img, lbl, is_lbl) in enumerate(zip(images, labels, is_labeled)):
        img_3d = np.squeeze(img, axis=0)
        sample = {
            'image': np.ascontiguousarray(img_3d),
            'label': np.ascontiguousarray(lbl),
            'is_labeled': is_lbl
        }
        paired_sample = None
        if batch_size > 1:
            pair_idx = pair_indices[idx]
            paired_img_3d = np.squeeze(images[pair_idx], axis=0)
            paired_sample = {
                'image': np.ascontiguousarray(paired_img_3d),
                'label': np.ascontiguousarray(labels[pair_idx]),
                'is_labeled': is_labeled[pair_idx]
            }
        sample = transforms.Compose([
            DualTransformWrapper(labeled_aug_in, unlabeled_aug_in, paired_sample),
            ToTensor()
        ])(sample)
        sampled_batch.append(sample)
        if controller is not None and 'aug_idx' in sample:
            aug_indices.append(sample['aug_idx'])
    if controller is not None:
        controller.record_batch(aug_indices)

    batch_dict = {
        'image': torch.stack([s['image'] for s in sampled_batch]),
        'label': torch.stack([s['label'] for s in sampled_batch]),
        'is_labeled': torch.tensor([s['is_labeled'] for s in sampled_batch], dtype=torch.bool)
    }
    return batch_dict