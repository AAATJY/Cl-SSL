"""
将区域感知对比学习（RACL）稳健地引入本训练脚本：
- 使用教师模型的解码器空间特征作为正样本，学生模型特征作为锚点，避免强增广带来的不稳定
- 延迟启用 + 线性warmup的对比权重
- 仅在边缘区域进行体素级对比、核心区域进行补丁级对比并进行采样控制，降低方差
- 每2500 iter动态调整阈值/权重/Top-K，稳定收敛
- 保持原有监督+一致性训练结构与参数基本不变

实测建议默认参数：
- contrast_weight=0.06~0.08（默认0.07）
- contrast_start_iter=3000
- patch_size=16, temp=0.1, hard_neg_k=32
加速版：引入AMP/TF32、DataLoader提速、RACL降采样与间隔计算、channels_last_3d与可选compile
"""

import argparse
import logging
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
from torch.cuda.amp import autocast, GradScaler
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataloaders.la_version1_3 import (
    LAHeart, ToTensor, TwoStreamBatchSampler
)
# 使用带对比学习分支的网络（包含RACL）
from networks.vnet_cl4 import VNet
from utils import ramps, losses
from utils.lossesplus import BoundaryLoss, FocalLoss  # 需在文件头部导入


class AugmentationController:
    def __init__(self, max_iter):
        self.iter = 0
        self.max_iter = max_iter
        self.current_strength = 0.1  # 初始增强强度

    def get_strength(self):
        return self.current_strength

    def step(self):
        self.iter = min(self.iter + 1, self.max_iter)
        self.current_strength = 0.1 + 0.4 * (self.iter / self.max_iter)


# MPL损失控制器（保持原逻辑）
class MPLController:
    def __init__(self, T=5, alpha=0.9, grad_scale=0.1):
        self.T = T
        self.alpha = alpha
        self.grad_scale = grad_scale
        self.student_loss_history = []
        self.current_trend = 0.0

    def compute_meta_grad(self, teacher_loss, student_params):
        teacher_loss.requires_grad_(True)
        grad_teacher = torch.autograd.grad(
            teacher_loss,
            student_params,
            create_graph=True,
            allow_unused=True
        )
        meta_grads = []
        for g_t, s_param in zip(grad_teacher, student_params):
            if g_t is None:
                meta_grads.append(None)
                continue
            grad_student = torch.autograd.grad(
                g_t.sum(),
                s_param,
                retain_graph=True,
                allow_unused=True
            )
            meta_grad = -self.grad_scale * (grad_student[0] if grad_student[0] is not None else 0.0)
            meta_grads.append(meta_grad)
        return meta_grads

    def get_teacher_weight(self):
        return torch.sigmoid(torch.tensor(self.current_trend))


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/ubuntu/workspace/Cl-SSL/data/2018LA_Seg_Training Set/',
                    help='Name of Experiment')
parser.add_argument('--exp', type=str, default='train_origin_5_fast_racl', help='model_name')
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
# ========== 新增：加速与对比学习参数 ==========
parser.add_argument('--amp', type=int, default=1, help='启用混合精度AMP(1/0)')
parser.add_argument('--tf32', type=int, default=1, help='启用TF32(1/0, 仅Ampere+)')
parser.add_argument('--compile', type=int, default=0, help='启用torch.compile(2.0+)，可能进一步加速')
parser.add_argument('--num_workers', type=int, default=8, help='DataLoader workers 数量')
# RACL控制（温和取样，降低计算量）
parser.add_argument('--contrast_weight', type=float, default=0.07, help='对比学习损失权重（小、稳）')
parser.add_argument('--contrast_start_iter', type=int, default=3000, help='启用对比学习的迭代次数（延迟启动）')
parser.add_argument('--contrast_interval', type=int, default=2, help='每隔多少iter计算一次RACL（>=1，2表示隔一次）')
parser.add_argument('--contrast_patch_size', type=int, default=16, help='对比学习补丁大小')
parser.add_argument('--contrast_temp', type=float, default=0.1, help='对比学习温度参数')
parser.add_argument('--contrast_hard_neg_k', type=int, default=32, help='对比学习hard negative数量')
# 降采样（减少开销），可按需调大以追求极致精度
parser.add_argument('--racl_core_patches', type=int, default=8, help='核心patch采样上限(默认8)')
parser.add_argument('--racl_edge_patches', type=int, default=12, help='边缘patch采样上限(默认12)')
parser.add_argument('--racl_voxels', type=int, default=128, help='边缘体素采样上限(默认128)')
# 是否仅对无标注样本做对比学习（省时；如需更强监督可设为0）
parser.add_argument('--contrast_unlabeled_only', type=int, default=1, help='仅无标注样本做RACL')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "/"

batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

# ======== 全局加速配置 ========
if args.tf32:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
else:
    # 允许cudnn自动选择最快算法（更快）
    cudnn.benchmark = True

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


if __name__ == "__main__":
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


    # ======== 模型创建（带RACL） ========
    def create_model(teacher=False):
        if teacher:
            net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm',
                       has_dropout=False, mc_dropout=True, mc_dropout_rate=args.mc_dropout_rate)
        else:
            net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
        # RACL参数（降采样以加速）
        net.contrast_learner.patch_size = args.contrast_patch_size
        net.contrast_learner.temp = args.contrast_temp
        net.contrast_learner.hard_neg_k = args.contrast_hard_neg_k
        net.contrast_learner.patch_sample_k = args.racl_core_patches
        net.contrast_learner.edge_patch_sample_k = args.racl_edge_patches
        net.contrast_learner.voxel_sample_k = args.racl_voxels
        model = net.cuda()
        # 使用channels_last_3d可进一步提速显存访问（PyTorch>=1.12）
        try:
            model = model.to(memory_format=torch.channels_last_3d)
        except Exception:
            pass
        return model


    student_model = create_model(teacher=False)
    teacher_model = create_model(teacher=True)
    teacher_model.load_state_dict(student_model.state_dict(), strict=False)

    # 可选：torch.compile 进一步加速（需要PyTorch>=2.0）
    if args.compile:
        try:
            student_model = torch.compile(student_model, mode="max-autotune")
            teacher_model = torch.compile(teacher_model, mode="max-autotune")
        except Exception as e:
            logging.info(f"torch.compile 不可用或失败，跳过。err={e}")

    teacher_optimizer = optim.SGD(teacher_model.parameters(), lr=base_lr * 0.1, momentum=0.9, weight_decay=0.0001)
    student_optimizer = optim.SGD(student_model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    # AMP scaler
    scaler_s = GradScaler(enabled=bool(args.amp))
    scaler_t = GradScaler(enabled=bool(args.amp))

    # 控制器
    mpl_controller = MPLController(T=10, alpha=0.95)
    meta_controller = MetaAugController(num_aug=6, init_temp=0.6,
                                        init_weights=[0.166, 0.166, 0.166, 0.166, 0.166, 0.166]).cuda()
    aug_controller = AugmentationController(args.max_iterations)

    # 数据与增强
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
        AugmentationFactory.strong_base_aug(patch_size),
    ])

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


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    # DataLoader 提速：更多worker、prefetch与持久worker
    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        worker_init_fn=worker_init_fn
    )

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

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    contrast_enabled = False

    log_interval = 10  # 降低日志频率，减少I/O开销

    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            if iter_num >= args.contrast_start_iter and not contrast_enabled:
                logging.info(f"启用区域感知对比学习 at iteration {iter_num}")
                contrast_enabled = True

            aug_controller.step()

            # 数据准备（非阻塞+channels_last_3d）
            sampled_batch = batch_aug_wrapper(sampled_batch, labeled_aug_in, unlabeled_aug_in, meta_controller)
            volume_batch = sampled_batch['image'].cuda(non_blocking=True)
            label_batch = sampled_batch['label'].cuda(non_blocking=True)
            try:
                volume_batch = volume_batch.to(memory_format=torch.channels_last_3d)
            except Exception:
                pass

            # ====== 教师伪标签（同视图，稳定） ======
            with torch.no_grad(), autocast(enabled=bool(args.amp)):
                t_seg_all, _, t_dec_feats = teacher_model(volume_batch, return_contrast_feats=False,
                                                          return_decoder_feats=True)
                teacher_logits_u = t_seg_all[labeled_bs:] / args.temperature
                probs = F.softmax(teacher_logits_u, dim=1)
                max_probs, _ = torch.max(probs, dim=1)
                threshold = args.base_threshold + (1 - args.base_threshold) * ramps.sigmoid_rampup(iter_num,
                                                                                                   max_iterations)
                mask = (max_probs > threshold).float().unsqueeze(1)

            # ====== 学生前向与损失（AMP） ======
            with autocast(enabled=bool(args.amp)):
                s_seg_all, _, s_dec_feats = student_model(volume_batch, return_contrast_feats=False,
                                                          return_decoder_feats=True)

                # 监督损失
                focal_criterion = FocalLoss(alpha=0.8, gamma=2)
                loss_seg = F.cross_entropy(s_seg_all[:labeled_bs], label_batch[:labeled_bs], label_smoothing=0.1)
                # Dice/Boundary 用FP32更稳
                outputs_soft = F.softmax(s_seg_all, dim=1).float()
                loss_seg_dice = losses.dice_loss(outputs_soft[:labeled_bs, 1], (label_batch[:labeled_bs] == 1))
                loss_boundary = BoundaryLoss()(outputs_soft[:labeled_bs, 1], (label_batch[:labeled_bs] == 1).float())
                loss_focal = focal_criterion(s_seg_all[:labeled_bs], label_batch[:labeled_bs])
                supervised_loss = 0.3 * (loss_seg + loss_seg_dice) + 0.4 * loss_boundary + 0.3 * loss_focal

                # 一致性损失
                consistency_weight = get_current_consistency_weight(iter_num // 150)
                consistency_dist = consistency_criterion(
                    s_seg_all[labeled_bs:],
                    label_smoothing(probs, factor=0.1)
                )
                weighted_loss = consistency_dist * mask
                masked_consistency = weighted_loss.view(weighted_loss.shape[0], -1).mean(dim=1)
                consistency_loss = consistency_weight * torch.mean(weighted_loss)

                # RACL（按间隔计算 + 仅无标注样本可选）
                weighted_contrast_loss = torch.tensor(0.0, device=volume_batch.device)
                if contrast_enabled and (iter_num % args.contrast_interval == 0):
                    contrast_loss = torch.tensor(0.0, device=volume_batch.device)
                    B = volume_batch.size(0)
                    for i in range(B):
                        if args.contrast_unlabeled_only and i < labeled_bs:
                            continue
                        anchor_feat = s_dec_feats[i].unsqueeze(0)  # 学生锚
                        positive_feat = t_dec_feats[i].unsqueeze(0)  # 教师正样本
                        if i < labeled_bs:
                            label_map = label_batch[i].unsqueeze(0).unsqueeze(1)
                            prob_map = None
                        else:
                            teacher_logits_i = t_seg_all[i] / args.temperature
                            probs_i = F.softmax(teacher_logits_i, dim=0)
                            max_conf, _ = torch.max(probs_i, dim=0)
                            pseudo_label = torch.argmax(probs_i, dim=0)
                            label_map = pseudo_label.unsqueeze(0).unsqueeze(1)
                            prob_map = max_conf.unsqueeze(0).unsqueeze(1)
                        contrast_loss = contrast_loss + student_model.contrast_learner(
                            anchor_feat, positive_feat, labels=label_map, prob_maps=prob_map
                        )
                    if args.contrast_unlabeled_only:
                        denom = max(1, (B - labeled_bs))
                    else:
                        denom = B
                    contrast_loss = contrast_loss / denom
                    contrast_weight = args.contrast_weight * min(1.0, (iter_num - args.contrast_start_iter) / 2000.0)
                    weighted_contrast_loss = contrast_weight * contrast_loss

                student_loss = supervised_loss + consistency_loss + weighted_contrast_loss

            # 学生更新（AMP安全剪裁）
            student_optimizer.zero_grad(set_to_none=True)
            if args.amp:
                scaler_s.scale(student_loss).backward()
                scaler_s.unscale_(student_optimizer)
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip)
                scaler_s.step(student_optimizer)
                scaler_s.update()
            else:
                student_loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip)
                student_optimizer.step()

            # 元增强权重更新（无需保留学生图）
            meta_controller.update_weights(masked_consistency.detach())

            # ====== 教师更新（AMP） ======
            with torch.no_grad():
                meta_labels = torch.softmax(s_seg_all.detach(), dim=1)

            with autocast(enabled=bool(args.amp)):
                teacher_outputs = teacher_model(volume_batch, return_contrast_feats=False)
                teacher_supervised_loss = F.cross_entropy(
                    teacher_outputs[:labeled_bs],
                    label_batch[:labeled_bs].long(),
                    label_smoothing=0.1,
                    reduction='mean'
                )
                teacher_consistency_loss = losses.softmax_kl_loss(
                    teacher_outputs[labeled_bs:], meta_labels[labeled_bs:]
                ).mean()
                teacher_weight = mpl_controller.get_teacher_weight()
                teacher_loss = teacher_supervised_loss + teacher_weight * teacher_consistency_loss

            teacher_optimizer.zero_grad(set_to_none=True)
            if args.amp:
                scaler_t.scale(teacher_loss).backward(retain_graph=True)
                # 元梯度叠加
                meta_grads = mpl_controller.compute_meta_grad(
                    teacher_loss=teacher_consistency_loss,
                    student_params=list(student_model.parameters())
                )
                for t_param, meta_g in zip(teacher_model.parameters(), meta_grads):
                    if meta_g is not None and t_param.grad is not None:
                        t_param.grad += meta_g.to(t_param.device)
                scaler_t.unscale_(teacher_optimizer)
                torch.nn.utils.clip_grad_norm_(teacher_model.parameters(), args.grad_clip)
                scaler_t.step(teacher_optimizer)
                scaler_t.update()
            else:
                teacher_loss.backward(retain_graph=True)
                meta_grads = mpl_controller.compute_meta_grad(
                    teacher_loss=teacher_consistency_loss,
                    student_params=list(student_model.parameters())
                )
                for t_param, meta_g in zip(teacher_model.parameters(), meta_grads):
                    if meta_g is not None and t_param.grad is not None:
                        t_param.grad += meta_g.to(t_param.device)
                torch.nn.utils.clip_grad_norm_(teacher_model.parameters(), args.grad_clip)
                teacher_optimizer.step()

            # 双向同步
            alpha_teacher = args.teacher_alpha
            with torch.no_grad():
                for t_param, s_param in zip(teacher_model.parameters(), student_model.parameters()):
                    t_param.data.mul_(alpha_teacher).add_(s_param.data, alpha=1 - alpha_teacher)
            update_ema_variables(teacher_model, student_model, alpha=0.999, global_step=iter_num)

            iter_num += 1
            if iter_num % log_interval == 0:
                writer.add_scalar('uncertainty/mask_per', torch.sum(mask) / mask.numel(), iter_num)
                writer.add_scalar('uncertainty/threshold', threshold, iter_num)
                writer.add_scalar('lr', lr_, iter_num)
                writer.add_scalar('loss/total', student_loss, iter_num)
                writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
                writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
                writer.add_scalar('train/consistency_loss', consistency_loss, iter_num)
                writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
                writer.add_scalar('train/consistency_dist', torch.mean(consistency_dist), iter_num)
                if contrast_enabled and (iter_num % args.contrast_interval == 0):
                    writer.add_scalar('loss/weighted_contrast_loss', weighted_contrast_loss, iter_num)
                logging.info('iter %d : loss=%.6f cons_w=%.4f contrast=%s' %
                             (iter_num, float(student_loss), float(consistency_weight), str(contrast_enabled)))


            # 余弦退火与RACL动态
            def adjust_learning_rate(optimizer, iteration, max_iter, base_lr):
                lr = base_lr * (math.cos(math.pi * iteration / max_iter) + 1) / 2
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr


            if iter_num % 2500 == 0:
                adjust_learning_rate(student_optimizer, iter_num, max_iterations, base_lr)
                adjust_learning_rate(teacher_optimizer, iter_num, max_iterations, base_lr * 0.1)
                if contrast_enabled:
                    epoch_ratio = iter_num / max_iterations
                    new_threshold = 0.42 + 0.15 * epoch_ratio
                    student_model.contrast_learner.edge_threshold = new_threshold
                    student_model.contrast_learner.loss_weights[0] = 1.0 - 0.25 * epoch_ratio
                    student_model.contrast_learner.loss_weights[1] = 0.7 + 0.25 * epoch_ratio
                    student_model.contrast_learner.hard_neg_k = int(32 + 8 * epoch_ratio)
                    logging.info(
                        f"[RACL] edge_thr={new_threshold:.3f}, weights={student_model.contrast_learner.loss_weights.tolist()}, topK={student_model.contrast_learner.hard_neg_k}")

            if iter_num % 1000 == 0:
                torch.save({
                    'student': student_model.state_dict(),
                    'teacher': teacher_model.state_dict(),
                    'contrast_learner': student_model.contrast_learner.state_dict(),
                    'iter_num': iter_num
                }, os.path.join(snapshot_path, f'iter_{iter_num}.pth'))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break

    torch.save({
        'student': student_model.state_dict(),
        'teacher': teacher_model.state_dict(),
        'contrast_learner': student_model.contrast_learner.state_dict(),
    }, os.path.join(snapshot_path, f'iter_{max_iterations}.pth'))
    writer.close()
