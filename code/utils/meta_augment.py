import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from dataloaders.la_version1_3 import (
    RandomRotFlip,RandomCrop,ElasticDeformation,GaussianBlur,ContrastAdjust,GammaCorrection,LocalShuffle,RandomNoise,RandomOcclusion,EdgeEnhancement,MotionArtifact,CutMix3D, MixUp3D,ToTensor
)


class MetaAugController(nn.Module):
    def __init__(self, num_aug, init_temp=0.1, init_weights=None):
        super().__init__()
        # 初始化权重参数
        if init_weights is not None:
            if len(init_weights) != num_aug:
                raise ValueError(f"init_weights长度({len(init_weights)})必须与num_aug({num_aug})一致")
            self.weights = nn.Parameter(torch.tensor(init_weights, dtype=torch.float32))
        else:
            self.weights = nn.Parameter(torch.ones(num_aug))

        # 其他参数
        self.temperature = init_temp
        self.optimizer = optim.Adam([self.weights], lr=1e-4)
        self.history = []

    def get_probs(self):
        """获取归一化后的增强概率分布（带温度系数）"""
        return F.softmax(self.weights / self.temperature, dim=-1)

    def record_batch(self, aug_indices):
        """记录批次增强选择"""
        self.history.append({
            'indices': aug_indices,
        })

    def update_weights(self, sample_loss):
        self.optimizer.zero_grad()
        grad_dict = {i: 0.0 for i in range(len(self.weights))}
        probs = self.get_probs()  # 获取增强策略的概率分布
        # 遍历每个样本
        for loss_value, record in zip(sample_loss, self.history):
            aug_indices = record['indices']  # 当前样本选择的增强策略索引
            loss_value = loss_value.unsqueeze(0)  # 确保损失是标量张量

            # 基于增强概率对损失进行加权
            weighted_loss = loss_value * probs[aug_indices].sum()

            # 计算该样本损失对权重的梯度
            grads = torch.autograd.grad(
                outputs=weighted_loss,
                inputs=self.weights,
                retain_graph=True,
                create_graph=False,
                allow_unused=True  # 允许未使用的参数
            )[0]
            # 对未选择的增强策略梯度置零
            for i in range(len(grads)):
                if i not in aug_indices:  # aug_indices 是被选择的增强策略索引
                    grads[i] = 0

            # 累计梯度到对应的增强策略索引
            for idx in aug_indices:
                grad_dict[idx] += grads[idx].item()

        # 更新权重
        if any(g != 0 for g in grad_dict.values()):
            grad_tensor = torch.tensor(
                [grad_dict[i] for i in range(len(self.weights))],
                device=self.weights.device
            )
            self.weights.grad = grad_tensor
            torch.nn.utils.clip_grad_norm_([self.weights], 5.0)
            self.optimizer.step()
            # **归一化权重**
        with torch.no_grad():
            self.weights.data = self.weights.data / self.weights.data.sum()  # 线性归一化
        # 清空历史记录
        self.history = []


# ------------------- 重构增强工厂 -------------------
class AugmentationFactory:
    # 基础固定增强（无需加权）
    @staticmethod
    def weak_base_aug(patch_size):
        return transforms.Compose([
            RandomRotFlip(p=0.5),
            RandomCrop(patch_size)
        ])

    @staticmethod
    def strong_base_aug(patch_size):
        return transforms.Compose([
            RandomRotFlip(p=0.8),
            ElasticDeformation(alpha=15),
            RandomCrop(patch_size)
        ])

    # 需要加权的增强方法池
    @staticmethod
    def get_weak_weighted_augs():
        return [
            # 增强后的参数
            GaussianBlur(sigma_range=(0.8, 1.8)),  # 增加模糊强度范围
            ContrastAdjust(factor_range=(0.5, 1.8)),  # 扩大对比度调整范围
            GammaCorrection(gamma_range=(0.5, 1.8)),  # 扩展伽马校正范围
            LocalShuffle(max_ratio=0.25, block_size=12),  # 增加局部打乱比例和块大小
            RandomNoise(sigma=0.18)  # 增加噪声强度
            # GaussianBlur(sigma_range=(0.5, 1.0)),
            # ContrastAdjust(factor_range=(0.7, 1.3)),
            # GammaCorrection(gamma_range=(0.7, 1.3)),
            # LocalShuffle(max_ratio=0.1, block_size=8),
            # RandomNoise(sigma=0.1)
        ]

    @staticmethod
    def get_strong_weighted_augs():
        return [
            # GaussianBlur(sigma_range=(1.5, 3.5)),
            # GammaCorrection(gamma_range=(0.3, 3.0)),
            # LocalShuffle(max_ratio=0.4, block_size=32),
            # EdgeEnhancement(),  # (需内部实现支持更强的锐化效果)
            # RandomNoise(sigma=0.4),
            # ContrastAdjust(factor_range=(0.6, 1.5))
            GaussianBlur(sigma_range=(1.0, 2.0)),
            GammaCorrection(gamma_range=(0.5, 2.5)),
            LocalShuffle(max_ratio=0.2, block_size=16),
            EdgeEnhancement(),
            RandomNoise(sigma=0.2),
            ContrastAdjust(factor_range=(0.8, 1.2)),
        ]


class WeightedWeakAugment(nn.Module):
    def __init__(self, aug_list, controller=None, alpha=1.0):
        super().__init__()
        self.augmenters = aug_list
        self.controller = controller  # 元控制器
        self.alpha = alpha  # 融合强度参数 (建议设置为1.0~1.5)
        self.default_weights = torch.tensor([0.15, 0.15, 0.2, 0.2, 0.3])
    def get_aug_weights(self):
        if self.controller is None:
            weights = self.default_weights
        else:
            weights = self.controller.weights
        weights_tensor = weights.clone().detach().float()
        weights_cpu = weights_tensor.cpu().numpy()
        return weights_cpu / np.sum(weights_cpu)

    def forward(self, sample,sample_pair=None):
        original = sample.copy()
        weights = self.get_aug_weights()
        aug_idx = np.random.choice(len(self.augmenters), p=weights)
        augmenter = self.augmenters[aug_idx]
        # 判断是否是CutMix/MixUp
        if isinstance(augmenter, (CutMix3D, MixUp3D)):
            if sample_pair is not None:
                aug_sample = augmenter(original.copy(), sample_pair)
            else:
                aug_sample = original
            # 不再二次Beta混合
        else:
            aug_sample = augmenter(original.copy())
            mix_ratio = np.random.beta(0.3 + self.alpha, 0.3 + (1 - self.alpha))
            aug_sample['image'] = (1 - mix_ratio) * original['image'] + mix_ratio * aug_sample['image']

        # 保留标签信息
        if 'label' in original:
            aug_sample['label'] = original['label']

        if self.controller is not None:
            aug_sample['aug_idx'] = aug_idx  # 添加索引信息
            return aug_sample
        else:
            return aug_sample


class DualTransformWrapper:
    def __init__(self, labeled_aug,unlabeled_aug, controller=None,paired_sample=None):
        self.labeled_aug = labeled_aug
        self.unlabeled_aug = unlabeled_aug
        self.controller = controller

    def __call__(self, sample,paired_sample=None):
        if sample.get('is_labeled', True):
            return self.labeled_aug(sample)
        else:
            # 判断unlabeled_aug是否需要paired_sample
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
        # 配对样本（仅为增强用）
        paired_sample = None
        if batch_size > 1:
            pair_idx = pair_indices[idx]
            paired_img_3d = np.squeeze(images[pair_idx], axis=0)
            paired_sample = {
                'image': np.ascontiguousarray(paired_img_3d),
                'label': np.ascontiguousarray(labels[pair_idx]),
                'is_labeled': is_labeled[pair_idx]
            }
        # 应用数据增强和转换为张量
        sample = transforms.Compose([
            DualTransformWrapper(labeled_aug_in, unlabeled_aug_in,paired_sample),
            ToTensor()
        ])(sample)
        sampled_batch.append(sample)
        if controller is not None and 'aug_idx' in sample:
            aug_indices.append(sample['aug_idx'])
    controller.record_batch(aug_indices)
    # 重组批次数据为字典
    batch_dict = {
        'image': torch.stack([s['image'] for s in sampled_batch]),
        'label': torch.stack([s['label'] for s in sampled_batch]),
        'is_labeled': torch.tensor([s['is_labeled'] for s in sampled_batch], dtype=torch.bool)
    }
    return batch_dict  # 返回字典类型