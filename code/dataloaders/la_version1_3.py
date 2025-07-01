# 该版本数据为通过元学习优化多扰动增强方法权重的数据增强文件，用于train_version1_4.py,train_version1_4_1.py,train_version1_4_2.py
import itertools

import h5py
import numpy as np
import torch
from scipy.ndimage import map_coordinates, gaussian_filter, rotate
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

class LAHeart(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None, labeled_idxs=None):
        self._base_dir = base_dir
        self.transform = transform
        self.labeled_idxs = labeled_idxs
        self.sample_list = []

        if split == 'train':
            with open(self._base_dir + '/../train.list', 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(self._base_dir + '/../test.list', 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n', '') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("Total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + "/" + image_name + "/mri_norm2.h5", 'r')
        image = np.ascontiguousarray(h5f['image'][:])  # 初始加载即保证连续
        label = np.ascontiguousarray(h5f['label'][:])
        sample = {'image': image, 'label': label}

        if self.labeled_idxs is not None:
            sample['is_labeled'] = idx in self.labeled_idxs
        if self.transform:
            sample = self.transform(sample)
        return sample

class RandomCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label ,is_labeled= sample['image'], sample['label'],  sample['is_labeled']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'image': image, 'label': label, 'is_labeled': is_labeled}

class GammaCorrection(object):
    """Gamma校正增强"""

    def __init__(self, gamma_range=(0.5, 2.0)):
        self.gamma_range = gamma_range

    def __call__(self, sample):
        image, label ,is_labeled= sample['image'], sample['label'],  sample['is_labeled']
        gamma = np.random.uniform(*self.gamma_range)

        # 保留原始数值范围
        min_val = image.min()
        max_val = image.max()
        epsilon = 1e-7  # 防止除零

        # 归一化后应用gamma校正
        normalized = (image - min_val) / (max_val - min_val + epsilon)
        corrected = np.power(normalized, gamma)

        # 恢复原始数值范围
        image = corrected * (max_val - min_val) + min_val
        return {'image': image, 'label': label, 'is_labeled': is_labeled}

class LocalShuffle(object):
    """局部像素随机扰动"""

    def __init__(self, max_ratio=0.1, block_size=8):
        self.max_ratio = max_ratio
        self.block_size = block_size

    def __call__(self, sample):
        image, label ,is_labeled= sample['image'], sample['label'],  sample['is_labeled']
        h, w, d = image.shape

        # 计算最大扰动块数
        total_voxels = h * w * d
        max_blocks = int(total_voxels * self.max_ratio / (self.block_size ** 3))

        for _ in range(np.random.randint(1, max_blocks + 1)):
            # 随机选择块位置
            i = np.random.randint(0, h - self.block_size)
            j = np.random.randint(0, w - self.block_size)
            k = np.random.randint(0, d - self.block_size)

            # 提取并打乱局部块
            block = image[i:i + self.block_size,
                    j:j + self.block_size,
                    k:k + self.block_size].copy()
            shuffled_block = np.random.permutation(block.flatten()).reshape(block.shape)

            # 应用扰动
            image[i:i + self.block_size,
            j:j + self.block_size,
            k:k + self.block_size] = shuffled_block

        return {'image': image, 'label': label, 'is_labeled': is_labeled}

class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label ,is_labeled= sample['image'], sample['label'],  sample['is_labeled']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2 * self.sigma,
                        2 * self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label, 'is_labeled': is_labeled}

class RandomRotFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if np.random.rand() > self.p:
            return sample

        image, label ,is_labeled= sample['image'], sample['label'],  sample['is_labeled']
        axes = np.random.choice(3, 2, replace=False)
        angle = np.random.choice([0, 90, 180, 270])

        # 3D旋转
        image = rotate(image, angle, axes=axes, reshape=False, mode='reflect')
        label = rotate(label, angle, axes=axes, reshape=False, mode='nearest')

        # 随机翻转
        if np.random.rand() > 0.5:
            flip_axis = np.random.randint(3)
            image = np.flip(image, flip_axis)
            label = np.flip(label, flip_axis)

        return {'image': image, 'label': label, 'is_labeled': is_labeled}

class EdgeEnhancement:
    def __call__(self, sample):
        image = sample['image']
        sobel_x = gaussian_filter(np.gradient(image, axis=0)[1], 1)
        sobel_y = gaussian_filter(np.gradient(image, axis=1)[1], 1)
        edge_mag = np.clip(np.sqrt(sobel_x**2 + sobel_y**2), 0, 1)
        image = np.clip(image + 0.3*edge_mag, 0, 1)
        return {'image': image, 'label': sample['label'], 'is_labeled': sample['is_labeled']}

class ElasticDeformation:
    # def __init__(self, alpha=10, sigma=3):

    def __init__(self, alpha=10, sigma=4):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        shape = image.shape
        dx = self.alpha * np.random.randn(*shape)
        dy = self.alpha * np.random.randn(*shape)
        dz = self.alpha * np.random.randn(*shape)

        dx = gaussian_filter(dx, self.sigma, mode='constant')
        dy = gaussian_filter(dy, self.sigma, mode='constant')
        dz = gaussian_filter(dz, self.sigma, mode='constant')

        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
        indices = (x + dx).reshape(-1, 1), (y + dy).reshape(-1, 1), (z + dz).reshape(-1, 1)

        image = map_coordinates(image, indices, order=1).reshape(shape)
        label = map_coordinates(label, indices, order=0).reshape(shape)
        return {'image': image, 'label': label, 'is_labeled': sample['is_labeled']}

class GaussianBlur:
    def __init__(self, sigma_range=(0.1, 2.0)):
        self.sigma_range = sigma_range

    def __call__(self, sample):
        image = sample['image']
        sigma = np.random.uniform(*self.sigma_range)
        image = gaussian_filter(image, sigma=sigma)
        return {'image': image, 'label': sample['label'], 'is_labeled': sample['is_labeled']}

class ContrastAdjust:
    def __init__(self, factor_range=(0.7, 1.3)):
        self.factor_range = factor_range

    def __call__(self, sample):
        image = sample['image']
        factor = np.random.uniform(*self.factor_range)
        mean_val = image.mean()
        return {'image': (image - mean_val) * factor + mean_val, 'label': sample['label'], 'is_labeled': sample['is_labeled']}

class RandomOcclusion:
    # def __init__(self, max_occlusion_size=32, num_occlusions=3):
    def __init__(self, max_occlusion_size=32, num_occlusions=5):
        self.max_size = max_occlusion_size
        self.num_occlusions = num_occlusions

    def __call__(self, sample):
        image = sample['image']
        h, w, d = image.shape

        for _ in range(np.random.randint(1, self.num_occlusions + 1)):
            oh = np.random.randint(1, self.max_size)
            ow = np.random.randint(1, self.max_size)
            od = np.random.randint(1, self.max_size // 2)

            x = np.random.randint(0, h - oh)
            y = np.random.randint(0, w - ow)
            z = np.random.randint(0, d - od)

            image[x:x + oh, y:y + ow, z:z + od] = np.random.normal(0, 0.1, (oh, ow, od))

        return {'image': image, 'label': sample['label'], 'is_labeled': sample['is_labeled']}

class MotionArtifact:
    def __init__(self, max_lines=5, intensity=0.8):
        self.max_lines = max_lines
        self.intensity = intensity

    def __call__(self, sample):
        image = sample['image']
        direction = np.random.choice(['x', 'y', 'z'])
        stripes = np.random.randint(1, self.max_lines + 1)

        for _ in range(stripes):
            stripe_width = np.random.randint(1, 3)
            intensity = np.random.normal(0, self.intensity)

            if direction == 'x':
                pos = np.random.randint(image.shape[0])
                image[pos:pos + stripe_width] += intensity
            elif direction == 'y':
                pos = np.random.randint(image.shape[1])
                image[:, pos:pos + stripe_width] += intensity
            else:
                pos = np.random.randint(image.shape[2])
                image[:, :, pos:pos + stripe_width] += intensity

        return {'image': np.clip(image, -1, 1), 'label': sample['label'], 'is_labeled': sample['is_labeled']}

class CutMix3D:
    def __init__(self, beta=1.0, prob=0.5):
        self.beta = beta
        self.prob = prob

    def __call__(self, sample, sample_pair=None):
        if np.random.rand() > self.prob:
            return sample  # 不做CutMix
        image, label = sample['image'], sample['label']

        # 如果没有提供配对样本，则直接返回
        if sample_pair is None:
            return sample

        image2, label2 = sample_pair['image'], sample_pair['label']

        D, H, W = image.shape
        # 采样CutMix块
        cut_rat = np.random.beta(self.beta, self.beta)
        cut_d = int(D * cut_rat)
        cut_h = int(H * cut_rat)
        cut_w = int(W * cut_rat)

        # 随机中心
        cd = np.random.randint(D)
        ch = np.random.randint(H)
        cw = np.random.randint(W)

        # 坐标范围
        d1 = np.clip(cd - cut_d // 2, 0, D)
        d2 = np.clip(cd + cut_d // 2, 0, D)
        h1 = np.clip(ch - cut_h // 2, 0, H)
        h2 = np.clip(ch + cut_h // 2, 0, H)
        w1 = np.clip(cw - cut_w // 2, 0, W)
        w2 = np.clip(cw + cut_w // 2, 0, W)

        # 替换块
        image_new = image.copy()
        label_new = label.copy()
        image_new[d1:d2, h1:h2, w1:w2] = image2[d1:d2, h1:h2, w1:w2]
        label_new[d1:d2, h1:h2, w1:w2] = label2[d1:d2, h1:h2, w1:w2]

        return {'image': image_new, 'label': label_new, 'is_labeled': sample['is_labeled']}

class MixUp3D:
    def __init__(self, alpha=0.4, prob=0.5):
        self.alpha = alpha
        self.prob = prob

    def __call__(self, sample, sample_pair=None):
        if np.random.rand() > self.prob:
            return sample  # 不做MixUp
        image, label = sample['image'], sample['label']
        if sample_pair is None:
            return sample
        image2, label2 = sample_pair['image'], sample_pair['label']

        lam = np.random.beta(self.alpha, self.alpha)
        image_new = lam * image + (1 - lam) * image2
        # 标签一为one-hot时可直接混合，否则视为soft label
        label_new = lam * label + (1 - lam) * label2

        return {'image': image_new, 'label': label_new, 'is_labeled': sample['is_labeled']}

class ToTensor(object):
    def __call__(self, sample):
        # 显式转换为连续数组
        image = np.ascontiguousarray(sample['image'])
        label = np.ascontiguousarray(sample['label'])

        image = image.reshape(1, *image.shape).astype(np.float32)

        output = {
            'image': torch.from_numpy(image),
            'label': torch.from_numpy(label).long(),
            'is_labeled': sample['is_labeled']
        }
        # 如果有 aug_idx 字段就保留
        if 'aug_idx' in sample:
            output['aug_idx'] = sample['aug_idx']
        return output


class TwoStreamBatchSampler(Sampler):
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)

def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    args = [iter(iterable)] * n
    return zip(*args)