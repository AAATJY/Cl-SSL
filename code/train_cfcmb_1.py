"""
并加入基于区域感知对比学习的 CFCMB（二分类版本）
"""

import argparse
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import math
from utils.meta_augment_2 import (
    MetaAugController, DualTransformWrapper, AugmentationFactory, WeightedWeakAugment, batch_aug_wrapper
)
import random
import shutil
import sys
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataloaders.la_version1_3 import (
    LAHeart, ToTensor, TwoStreamBatchSampler
)
from networks.vnet_cfcmb import VNet
from utils import ramps, losses
from utils.lossesplus import BoundaryLoss, FocalLoss
from utils.cmcfb import RegionContrastiveMemoryBinary
class AugmentationController:
    def __init__(self, max_iter):
        self.iter = 0
        self.max_iter = max_iter
        # 新增动态增强参数
        self.current_strength = 0.1  # 初始增强强度

    def get_strength(self):
        """动态增强强度"""
        return self.current_strength

    def step(self):
        self.iter = min(self.iter + 1, self.max_iter)
        # 线性增强策略
        self.current_strength = 0.1 + 0.4 * (self.iter / self.max_iter)


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/zlj/workspace/tjy/MeTi-SSL/data/2018LA_Seg_Training Set/', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='train_cfcmb_1', help='model_name')
parser.add_argument('--max_iterations', type=int, default=15000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
### costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float, default=0.4, help='伪标签温度缩放')
parser.add_argument('--base_threshold', type=float, default=0.7, help='基础置信度阈值')
parser.add_argument('--mc_dropout_rate', type=float, default=0.2, help='MC Dropout概率')
parser.add_argument('--meta_grad_scale', type=float, default=0.1, help='元梯度缩放系数')
parser.add_argument('--grad_clip', type=float, default=3.0, help='梯度裁剪阈值')
parser.add_argument('--teacher_alpha', type=float, default=0.99, help='教师模型EMA系数')
parser.add_argument('--queue_size', type=int, default=200, help='CFCMB 每类队列长度')
parser.add_argument('--beta_con', type=float, default=0.1, help='有标注对比损失权重')
parser.add_argument('--lambda_con', type=float, default=0.3, help='无标注对比损失权重')
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
    """标签平滑函数"""
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
    meta_controller = MetaAugController(num_aug=6, init_temp=0.6,
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
    # 初始化 CFCMB（特征维度取 student decoder 返回的通道数）
    # 要与 VNet 中 block_nine 输出通道数一致（n_filters），你可根据 init 时 n_filters 设置修改 feat_dim
    feat_dim = 16  # 如果你初始化 VNet 的 n_filters!=16，请改此处
    cfcmb = RegionContrastiveMemoryBinary(feat_dim=feat_dim, queue_size=args.queue_size, tau=0.1, device="cuda")

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr

    for epoch_num in tqdm(range(max_epoch), ncols=70):
        for i_batch, sampled_batch in enumerate(trainloader):
            # ========== 动态增强控制==========
            aug_controller.step()
            current_strength = aug_controller.get_strength()  # 获取当前增强强度
            time2 = time.time()
            # ========== 数据准备 ==========
            weak_volume = sampled_batch['image'].cuda()
            weak_volume_batch = weak_volume[labeled_bs:]
            sampled_batch = batch_aug_wrapper(sampled_batch, labeled_aug_in, unlabeled_aug_in, meta_controller)
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            # ========== 阶段1：教师模型生成伪标签==========
            with torch.no_grad():
                teacher_outputs = teacher_model(weak_volume_batch, return_features=False, enable_dropout=True)
                teacher_outputs = teacher_outputs / args.temperature
                probs = F.softmax(teacher_outputs, dim=1)
                max_probs, _ = torch.max(probs, dim=1)
                threshold = args.base_threshold + (1 - args.base_threshold) * ramps.sigmoid_rampup(iter_num,
                                                                                                   max_iterations)
                mask = (max_probs > threshold).float().unsqueeze(1)
                pseudo_labels = torch.argmax(probs, dim=1)
            # ========== 阶段2：学生模型 forward（返回 logits 和 features）==========
            student_outputs, student_features = student_model(volume_batch, return_features=True,enable_dropout=False)

            # ========== 监督损失（有标注部分）==========
            focal_criterion = FocalLoss(alpha=0.8, gamma=2)
            loss_seg = F.cross_entropy(student_outputs[:labeled_bs], label_batch[:labeled_bs], label_smoothing=0.1)
            outputs_soft = F.softmax(student_outputs, dim=1)
            loss_seg_dice = losses.dice_loss(outputs_soft[:labeled_bs, 1, :, :, :],
                                             (label_batch[:labeled_bs] == 1).float())
            loss_boundary = BoundaryLoss()(outputs_soft[:labeled_bs, 1], (label_batch[:labeled_bs] == 1).float())
            loss_focal = focal_criterion(student_outputs[:labeled_bs], label_batch[:labeled_bs])
            supervised_loss = 0.3 * (loss_seg + loss_seg_dice) + 0.4 * loss_boundary + 0.3 * loss_focal

            # ========== 一致性损失（无标注部分，带动态 mask）==========
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            consistency_dist = consistency_criterion(
                student_outputs[labeled_bs:],
                label_smoothing(probs, factor=0.1)
            )
            weighted_loss = consistency_dist * mask  # shape [U_bs, C, D, H, W] or depending on loss
            # 取均值（保留和原一致性形式）
            consistency_loss = consistency_weight * torch.mean(weighted_loss)

            labeled_centers = cfcmb.compute_region_centers_3d(student_features[:labeled_bs], label_batch[:labeled_bs], num_classes=2)
            # 更新 CFCMB（仅在有标注处更新）
            cfcmb.update_from_centers(labeled_centers)

            # ========== 区域感知对比损失（有标注） ==========
            Lcon_sup = cfcmb.contrastive_loss_for_centers(labeled_centers)
            # ========== 无标注部分的区域中心（基于伪标签，不更新 CFCMB） ==========
            unlabeled_features = student_features[labeled_bs:]  # [U_bs, C_feat, D, H, W]
            unlabeled_pseudo = pseudo_labels  # [U_bs, D, H, W]
            unlabeled_centers = cfcmb.compute_region_centers_3d(unlabeled_features, unlabeled_pseudo, num_classes=2)
            Lcon_unsup = cfcmb.contrastive_loss_for_centers(unlabeled_centers)

            # ========== 总损失组合 ==========
            total_loss = supervised_loss + consistency_loss + args.beta_con * Lcon_sup + args.lambda_con * Lcon_unsup

            # ========== 优化步骤 ==========
            student_optimizer.zero_grad()
            total_loss.backward()
            meta_controller.update_weights(  # 保持你原来的元控制器调用
                (weighted_loss.view(weighted_loss.shape[0], -1).mean(dim=1) if weighted_loss.numel() else torch.zeros(
                    1).cuda())
            )
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip)
            student_optimizer.step()

            # EMA 更新教师
            update_ema_variables(student_model, teacher_model, alpha=args.teacher_alpha, global_step=iter_num)

            # ========== logging & 保存 ==========
            iter_num += 1
            writer.add_scalar('uncertainty/mask_per', torch.sum(mask) / mask.numel(), iter_num)
            writer.add_scalar('uncertainty/threshold', threshold, iter_num)
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/total_loss', total_loss, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('train/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('train/Lcon_sup', Lcon_sup, iter_num)
            writer.add_scalar('train/Lcon_unsup', Lcon_unsup, iter_num)

            logging.info('iteration %d : total_loss: %f cons_w: %f Lcon_sup: %f Lcon_unsup: %f' %
                         (iter_num, total_loss.item(), consistency_weight, float(Lcon_sup), float(Lcon_unsup)))


            # lr 调整（余弦退火）
            def adjust_learning_rate(optimizer, iteration, max_iter, base_lr):
                lr = base_lr * (math.cos(math.pi * iteration / max_iter) + 1) / 2
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr


            if iter_num % 2500 == 0:
                adjust_learning_rate(student_optimizer, iter_num, max_iterations, base_lr)
                adjust_learning_rate(teacher_optimizer, iter_num, max_iterations, base_lr * 0.1)

            if iter_num % 1000 == 0:
                torch.save({
                    'student': student_model.state_dict(),
                    'teacher': teacher_model.state_dict(),
                    'cfcmb': {k: list(v) for k, v in cfcmb.memory.items()}
                }, os.path.join(snapshot_path, f'iter_{iter_num}.pth'))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            break

            # 最终保存
        torch.save({
            'student': student_model.state_dict(),
            'teacher': teacher_model.state_dict(),
            'cfcmb': {k: list(v) for k, v in cfcmb.memory.items()}
        }, os.path.join(snapshot_path, f'iter_{max_iterations}.pth'))
        writer.close()