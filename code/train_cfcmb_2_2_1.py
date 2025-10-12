"""
MetaAugController  init_temp：0.6→0.4 提升探索；或在前1000 iter 固定增强，之后再让 meta 控制器更新权重，降低早期不稳定。
"""

import argparse
import logging
import os

from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import math
from utils.meta_augment import (
    MetaAugController, DualTransformWrapper, AugmentationFactory, WeightedWeakAugment, batch_aug_wrapper
)
import random
import shutil
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloaders.la_version1_3 import (
    LAHeart, ToTensor, TwoStreamBatchSampler
)
from networks.vnet_cfcmb_2 import VNet
from utils import ramps, losses
from utils.lossesplus import BoundaryLoss, FocalLoss

# [AMR-CMB] MOD: 引入多尺度对比记忆库
from utils.cmcfb_2 import AdaptiveMultiScaleRegionContrastiveMemory  # [AMR-CMB] MOD


class AugmentationController:
    def __init__(self, max_iter):
        self.iter = 0
        self.max_iter = max_iter
        self.current_strength = 0.1

    def get_strength(self):
        return self.current_strength

    def step(self):
        self.iter = min(self.iter + 1, self.max_iter)
        self.current_strength = 0.1 + 0.4 * (self.iter / self.max_iter)


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/zlj/workspace/tjy/MeTi-SSL/data/2018LA_Seg_Training Set/', help='Name of Experiment')
# parser.add_argument('--root_path', type=str, default='/root/autodl-tmp/Cl-SSL/data/2018LA_Seg_Training Set/', help='Name of Experiment')

parser.add_argument('--exp', type=str, default='train_cfcmb_2_2_123weak_MP_962', help='model_name')
parser.add_argument('--max_iterations', type=int, default=18000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
# consistency
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float, default=0.4, help='伪标签温度缩放')
parser.add_argument('--base_threshold', type=float, default=0.7, help='基础置信度阈值')
parser.add_argument('--mc_dropout_rate', type=float, default=0.2, help='MC Dropout概率')
parser.add_argument('--grad_clip', type=float, default=3.0, help='梯度裁剪阈值')
parser.add_argument('--teacher_alpha', type=float, default=0.99, help='教师模型EMA系数')
parser.add_argument('--queue_size', type=int, default=200, help='每类队列长度')
parser.add_argument('--beta_con', type=float, default=0.1, help='有标注对比损失权重')
parser.add_argument('--lambda_con', type=float, default=0.3, help='无标注对比损失权重')

# [AMR-CMB] MOD: 新增 AMR 参数
parser.add_argument('--amr_scales', type=str, default='enc3,enc4,dec', help='使用的多尺度键（逗号分隔）')
parser.add_argument('--amr_base_tau', type=float, default=0.15, help='AMR 基础温度')
parser.add_argument('--amr_conf_threshold', type=float, default=0.962, help='AMR 置信度阈值（用于可选伪标注更新）')
parser.add_argument('--amr_use_unlabeled_update', type=int, default=1, help='是否用高置信伪标注更新内存 0/1')
parser.add_argument('--amr_max_neg_per_class', type=int, default=128, help='每类负样本最大采样数')

args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "/"

batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    # np.random.seed(args.seed)
    np.random.seed()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (112, 112, 80)

def label_smoothing(labels, factor=0.1):
    return labels * (1 - factor) + factor / labels.size(1)

def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def create_model(ema=False, teacher=False):
    if teacher:
        net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm',
                   has_dropout=False, mc_dropout=True, mc_dropout_rate=args.mc_dropout_rate)
    else:
        net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
            param.requires_grad_(False)
    return model

def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)

if __name__ == "__main__":
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # 初始化模型、优化器
    student_model = create_model(teacher=False)
    teacher_model = create_model(teacher=True)
    teacher_model.load_state_dict(student_model.state_dict(), strict=False)

    teacher_optimizer = optim.SGD(teacher_model.parameters(), lr=base_lr * 0.1, momentum=0.9, weight_decay=0.0001)
    student_optimizer = optim.SGD(student_model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    # 控制器与增强器
    meta_controller = MetaAugController(num_aug=6, init_temp=0.4,
                                        init_weights=[0.166, 0.166, 0.166, 0.166, 0.166, 0.166]).cuda()
    aug_controller = AugmentationController(args.max_iterations)

    labeled_aug_in = transforms.Compose([
        WeightedWeakAugment(AugmentationFactory.get_weak_weighted_augs())
    ])
    labeled_aug_out = transforms.Compose([
        AugmentationFactory.weak_base_aug(patch_size),
    ])
    unlabeled_aug_in = transforms.Compose([
        WeightedWeakAugment(
            AugmentationFactory.get_strong_weighted_augs(),
            controller=meta_controller
        )
    ])
    unlabeled_aug_out = transforms.Compose([
        AugmentationFactory.weak_base_aug(patch_size),
    ])

    # dataloader
    labeled_idxs = list(range(16))
    unlabeled_idxs = list(range(16, 80))
    db_train = LAHeart(
        base_dir=train_data_path,
        split='train',
        transform=transforms.Compose([
            DualTransformWrapper(labeled_aug_out, unlabeled_aug_out),
            ToTensor()
        ]),
        labeled_idxs=labeled_idxs
    )
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    student_model.train()
    teacher_model.train()

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    # [AMR-CMB] MOD: 惰性初始化 AMR-CMB（首个 batch 前向后根据实际通道构造）
    amr = None  # type:Optional[AdaptiveMultiScaleRegionContrastiveMemory]

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr

    def compute_unlabeled_confidence(max_probs, pseudo_labels):
        U = max_probs.shape[0]
        scores = []
        for i in range(U):
            fg_mask = (pseudo_labels[i] == 1)
            if fg_mask.any():
                val = max_probs[i][fg_mask].mean()
            else:
                val = max_probs[i].mean()
            scores.append(val)
        return torch.stack(scores, dim=0)

    for epoch_num in tqdm(range(max_epoch)):
        for i_batch, sampled_batch in enumerate(trainloader):
            aug_controller.step()

            # 数据准备（弱增强给 teacher 产生伪标签）
            weak_volume = sampled_batch['image'].cuda()
            weak_volume_batch = weak_volume[labeled_bs:]
            sampled_batch = batch_aug_wrapper(sampled_batch, labeled_aug_in, unlabeled_aug_in, meta_controller)
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            # 教师模型伪标签（带温度与动态阈值）
            with torch.no_grad():
                teacher_outputs = teacher_model(weak_volume_batch, return_features=False, enable_dropout=True)
                teacher_outputs = teacher_outputs / args.temperature
                probs = F.softmax(teacher_outputs, dim=1)
                max_probs, _ = torch.max(probs, dim=1)
                threshold = args.base_threshold + (1 - args.base_threshold) * ramps.sigmoid_rampup(iter_num, max_iterations)
                mask = (max_probs > threshold).float().unsqueeze(1)
                pseudo_labels = torch.argmax(probs, dim=1)  # [U_bs,D,H,W]
                # [AMR-CMB] MOD: 无标注置信度向量
                unlabeled_conf_vec = compute_unlabeled_confidence(max_probs.detach(), pseudo_labels.detach())

            # 学生模型前向（多尺度特征）
            # [AMR-CMB] MOD: 开启 return_multi_scale=True
            student_outputs_all = student_model(volume_batch, return_features=True, enable_dropout=False, return_multi_scale=True)  # [AMR-CMB] MOD
            student_outputs, student_feat_last, student_feats_ms = student_outputs_all  # logits, x9, dict[str]->[B,C,D,H,W]

            # 惰性初始化 AMR（首个 batch 确定各尺度通道维度）
            if amr is None:
                use_scales = [s.strip() for s in args.amr_scales.split(',') if s.strip() in student_feats_ms]
                feat_dims = {s: student_feats_ms[s].shape[1] for s in use_scales}
                amr = AdaptiveMultiScaleRegionContrastiveMemory(
                    feat_dims=feat_dims,
                    queue_size=args.queue_size,
                    base_tau=args.amr_base_tau,
                    device="cuda",
                    scale_weights=None,  # 均分
                    adaptive_temp=True,
                    confidence_threshold=args.amr_conf_threshold,
                    class_balancing=True,
                    max_neg_per_class=args.amr_max_neg_per_class
                )
                logging.info(f"[AMR-CMB] Initialized with scales {feat_dims}")

            # 监督损失（有标注）
            focal_criterion = FocalLoss(alpha=0.8, gamma=2)
            loss_seg = F.cross_entropy(student_outputs[:labeled_bs], label_batch[:labeled_bs], label_smoothing=0.1)
            outputs_soft = F.softmax(student_outputs, dim=1)
            loss_seg_dice = losses.dice_loss(outputs_soft[:labeled_bs, 1, :, :, :],
                                             (label_batch[:labeled_bs] == 1).float())
            loss_boundary = BoundaryLoss()(outputs_soft[:labeled_bs, 1], (label_batch[:labeled_bs] == 1).float())
            loss_focal = focal_criterion(student_outputs[:labeled_bs], label_batch[:labeled_bs])
            supervised_loss = 0.3 * (loss_seg + loss_seg_dice) + 0.4 * loss_boundary + 0.3 * loss_focal

            # 一致性损失（无标注，带动态 mask）
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            consistency_dist = consistency_criterion(
                student_outputs[labeled_bs:],
                label_smoothing(probs, factor=0.1)
            )
            weighted_loss = consistency_dist * mask
            consistency_loss = consistency_weight * torch.mean(weighted_loss)

            # [AMR-CMB] MOD: 多尺度区域中心（有标注部分，更新内存）
            labeled_feats_dict = {s: f[:labeled_bs] for s, f in student_feats_ms.items() if s in amr.scales}
            labeled_centers_ms = amr.compute_multi_scale_region_centers(labeled_feats_dict, label_batch[:labeled_bs], num_classes=2)
            amr.update_from_centers(labeled_centers_ms, confidence_scores=None)  # 仅标注更新

            # [AMR-CMB] MOD: 有标注对比损失（多尺度加权）
            Lcon_sup, sup_scale_loss, sup_scale_class_loss = amr.contrastive_loss_all_scales(labeled_centers_ms)

            # [AMR-CMB] MOD: 无标注区域中心（基于伪标签，不更新或可选高置信更新）
            unlabeled_feats_dict = {s: f[labeled_bs:] for s, f in student_feats_ms.items() if s in amr.scales}
            unlabeled_centers_ms = amr.compute_multi_scale_region_centers(unlabeled_feats_dict, pseudo_labels, num_classes=2)

            if args.amr_use_unlabeled_update:
                # 将同一置信度向量用于所有尺度（也可按尺度单独估计）
                conf_dict = {s: unlabeled_conf_vec for s in amr.scales}
                amr.update_from_centers(unlabeled_centers_ms, confidence_scores=conf_dict)

            Lcon_unsup, unsup_scale_loss, unsup_scale_class_loss = amr.contrastive_loss_all_scales(unlabeled_centers_ms)

            # 总损失
            total_loss = supervised_loss + consistency_loss + args.beta_con * Lcon_sup + args.lambda_con * Lcon_unsup

            # 优化
            student_optimizer.zero_grad()
            total_loss.backward()
            meta_controller.update_weights(
                (weighted_loss.view(weighted_loss.shape[0], -1).mean(dim=1) if weighted_loss.numel() else torch.zeros(1).cuda())
            )
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip)
            student_optimizer.step()

            # EMA teacher
            update_ema_variables(student_model, teacher_model, alpha=args.teacher_alpha, global_step=iter_num)

            # logging
            iter_num += 1
            writer.add_scalar('uncertainty/mask_per', torch.sum(mask) / mask.numel(), iter_num)
            writer.add_scalar('uncertainty/threshold', threshold, iter_num)
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/total_loss', total_loss, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('train/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('train/Lcon_sup_total', Lcon_sup, iter_num)
            writer.add_scalar('train/Lcon_unsup_total', Lcon_unsup, iter_num)
            # [AMR-CMB] MOD: 记录多尺度损失
            for s, v in sup_scale_loss.items():
                writer.add_scalar(f'train/Lcon_sup_{s}', v, iter_num)
            for s, v in unsup_scale_loss.items():
                writer.add_scalar(f'train/Lcon_unsup_{s}', v, iter_num)

            logging.info(
                f'iteration {iter_num} : total_loss: {total_loss.item():.5f} '
                f'cons_w: {consistency_weight:.4f} '
                f'Lcon_sup: {float(Lcon_sup):.5f} Lcon_unsup: {float(Lcon_unsup):.5f}'
            )

            # lr 调整（余弦退火）
            def adjust_learning_rate(optimizer, iteration, max_iter, base_lr):
                lr = base_lr * (math.cos(math.pi * iteration / max_iter) + 1) / 2
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            if iter_num % 2500 == 0:
                adjust_learning_rate(student_optimizer, iter_num, max_iterations, base_lr)
                adjust_learning_rate(teacher_optimizer, iter_num, max_iterations, base_lr * 0.1)

            if iter_num % 1000 == 0:
                # [AMR-CMB] MOD: 序列化多尺度内存
                amr_state = None
                if amr is not None:
                    amr_state = {
                        scale: {k: [t.tolist() for t in v] for k, v in amr.memory[scale].items()}
                        for scale in amr.scales
                    }
                torch.save({
                    'student': student_model.state_dict(),
                    'teacher': teacher_model.state_dict(),
                    'amr_memory': amr_state
                }, os.path.join(snapshot_path, f'iter_{iter_num}.pth'))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            break

    # 最终保存
    amr_state = None
    if amr is not None:
        amr_state = {
            scale: {k: [t.tolist() for t in v] for k, v in amr.memory[scale].items()}
            for scale in amr.scales
        }
    torch.save({
        'student': student_model.state_dict(),
        'teacher': teacher_model.state_dict(),
        'amr_memory': amr_state
    }, os.path.join(snapshot_path, f'iter_{max_iterations}.pth'))
    writer.close()