"""
RCPS对比
"""

import argparse
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import math
from utils.meta_augment_2 import (
    MetaAugController, DualTransformWrapper, AugmentationFactory, WeightedWeakAugment,batch_aug_wrapper
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
from networks.vnet_cl3 import VNet
from utils import ramps, losses
from utils.lossesplus import BoundaryLoss, FocalLoss  # 需在文件头部导入

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

# 🆕 新增MPL损失控制器
class MPLController:
    def __init__(self, T=5, alpha=0.9,grad_scale=0.1):
        self.T = T  # 平滑窗口大小
        self.alpha = alpha  # 指数平滑系数
        self.grad_scale = grad_scale
        self.student_loss_history = []
        self.current_trend = 0.0  # 学生模型性能变化趋势

    def compute_meta_grad(self, teacher_loss, student_params):
        """计算元梯度（修正参数签名）"""
        # 确保teacher_loss需要梯度
        teacher_loss.requires_grad_(True)

        # 计算教师参数的一阶梯度
        grad_teacher = torch.autograd.grad(
            teacher_loss,
            student_params,  # 这里应为教师模型的参数
            create_graph=True,
            allow_unused=True
        )

        meta_grads = []
        for g_t, s_param in zip(grad_teacher, student_params):
            if g_t is None:
                meta_grads.append(None)
                continue
            # 计算二阶导数
            grad_student = torch.autograd.grad(
                g_t.sum(),
                s_param,
                retain_graph=True,
                allow_unused=True
            )
            meta_grad = -self.grad_scale * (grad_student[0] if grad_student[0] is not None else 0.0)
            meta_grads.append(meta_grad)
        return meta_grads

    def update(self, student_loss):
        """更新学生模型损失趋势"""
        self.student_loss_history.append(student_loss)
        if len(self.student_loss_history) > self.T:
            self.student_loss_history.pop(0)

        # 计算趋势变化
        if len(self.student_loss_history) >= 2:
            delta = self.student_loss_history[-2] - self.student_loss_history[-1]  # 损失下降为正值
            self.current_trend = self.alpha * self.current_trend + (1 - self.alpha) * delta

    def get_teacher_weight(self):
        """生成教师模型损失权重"""
        return torch.sigmoid(torch.tensor(self.current_trend))  # 趋势越好，权重越大

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/zlj/workspace/tjy/MeTi-SSL/data/2018LA_Seg_Training Set/', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='train_cl3', help='model_name')
parser.add_argument('--max_iterations', type=int, default=15000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=1, help='labeled_batch_size per gpu')
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
# 新增对比学习参数
parser.add_argument('--contrast_weight', type=float, default=0.1, help='对比学习损失权重')
parser.add_argument('--contrast_start_iter', type=int, default=2000, help='启用对比学习的迭代次数')
parser.add_argument('--contrast_patch_size', type=int, default=16, help='对比学习补丁大小')
parser.add_argument('--contrast_temp', type=float, default=0.1, help='对比学习温度参数')
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


if __name__ == "__main__":
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


    # 🆕 修改模型创建部分
    def create_model(ema=False, teacher=False):
        """创建模型并设置梯度状态"""
        # 网络结构差异
        if teacher:  # 教师模型加深结构
            net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm',
                       has_dropout=False,mc_dropout=True, mc_dropout_rate=args.mc_dropout_rate,)
        else:  # 学生模型基础结构
            net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
        model = net.cuda()
        # 梯度设置
        if ema:
            for param in model.parameters():
                param.detach_()  # 分离计算图
                param.requires_grad_(False)  # 显式禁用梯度
        return model

    # ================= 模型及优化器初始化 =================
    student_model = create_model(teacher=False)  # 可训练学生模型
    teacher_model = create_model(teacher=True)  # 可训练教师模型
    teacher_model.load_state_dict(student_model.state_dict(), strict=False)  # 关键修复

    # 设置对比学习模块参数
    student_model.contrast_learner.patch_size = args.contrast_patch_size
    student_model.contrast_learner.temp = args.contrast_temp
    teacher_model.contrast_learner.patch_size = args.contrast_patch_size
    teacher_model.contrast_learner.temp = args.contrast_temp
    # 设置对比学习模块参数

    teacher_optimizer = optim.SGD(teacher_model.parameters(), lr=base_lr * 0.1, momentum=0.9, weight_decay=0.0001)
    student_optimizer = optim.SGD(student_model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    # ================= MPL控制器、元控制器和增强控制器初始化 =================
    mpl_controller = MPLController(T=10, alpha=0.95)  # 初始化MPL控制器
    meta_controller = MetaAugController(num_aug=6,init_temp=0.6,init_weights=[0.166, 0.166, 0.166, 0.166, 0.166, 0.166]).cuda()
    aug_controller = AugmentationController(args.max_iterations)
    # ================= 增强策略及数据加载 =================
    labeled_aug_in = transforms.Compose([
        WeightedWeakAugment(AugmentationFactory.get_weak_weighted_augs())
    ])

    # 为对比学习创建不同的增强策略
    labeled_aug_weak = transforms.Compose([
        WeightedWeakAugment(AugmentationFactory.get_weak_weighted_augs())
    ])

    labeled_aug_strong = transforms.Compose([
        WeightedWeakAugment(AugmentationFactory.get_strong_weighted_augs())
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

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    # 对比学习启用标志
    contrast_enabled = False
    # ================= 训练循环 =================
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            # 检查是否启用对比学习
            if iter_num >= args.contrast_start_iter and not contrast_enabled:
                logging.info(f"启用对比学习 at iteration {iter_num}")
                contrast_enabled = True
            # ================= 动态增强控制 =================
            aug_controller.step()
            current_strength = aug_controller.get_strength()  # 获取当前增强强度
            time2 = time.time()
            # ================= 数据准备 =================
            weak_volume_batch = sampled_batch['image'].cuda()
            sampled_batch = batch_aug_wrapper(sampled_batch, labeled_aug_in, unlabeled_aug_in,meta_controller)
            strong_volume_batch = sampled_batch['image'].cuda()
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[labeled_bs:]

            # ========== 阶段1：教师模型生成伪标签 ==========
            with torch.no_grad():
                T = 8  # 增强次数
                aug_preds = []
                # 噪声扰动增强
                for _ in range(T // 2):
                    noise = torch.randn_like(unlabeled_volume_batch) * current_strength
                    aug_inputs = unlabeled_volume_batch + noise
                    aug_preds.append(teacher_model(aug_inputs)[0])

                # 3D旋转增强（修正版本）
                for _ in range(T // 2):
                    angle = random.uniform(-10, 10) * current_strength
                    theta = torch.zeros((unlabeled_volume_batch.size(0), 3, 4),
                                        device=unlabeled_volume_batch.device)
                    theta[:, 0, 0] = np.cos(np.radians(angle))
                    theta[:, 0, 1] = -np.sin(np.radians(angle))
                    theta[:, 1, 0] = np.sin(np.radians(angle))
                    theta[:, 1, 1] = np.cos(np.radians(angle))
                    theta[:, 2, 2] = 1.0

                    grid = F.affine_grid(theta, unlabeled_volume_batch.size(), align_corners=False)
                    aug_inputs = F.grid_sample(
                        unlabeled_volume_batch,
                        grid,
                        mode='bilinear',
                        padding_mode='zeros',
                        align_corners=False
                    )
                    aug_preds.append(teacher_model(aug_inputs)[0])

                # 集成预测结果
                teacher_outputs = torch.stack(aug_preds).mean(dim=0)
                teacher_outputs = teacher_outputs / args.temperature  # 温度缩放

                # 动态置信度阈值
                probs = F.softmax(teacher_outputs, dim=1)
                max_probs, _ = torch.max(probs, dim=1)
                threshold = args.base_threshold + (1 - args.base_threshold) * ramps.sigmoid_rampup(iter_num,max_iterations)
                mask = (max_probs > threshold).float().unsqueeze(1)  # 保持维度对齐

            # ========== 阶段2：学生模型训练 ==========
            # 原始视图的分割输出
            student_seg_out, student_contrast_feats = student_model(volume_batch, return_contrast_feats=True)

            # 弱增强视图的特征 (用于对比学习的锚点)
            _, weak_contrast_feats = student_model(weak_volume_batch, return_contrast_feats=True)

            # 强增强视图的特征 (用于对比学习的正样本)
            with torch.no_grad():
                _, strong_contrast_feats = student_model(strong_volume_batch, return_contrast_feats=True)
                strong_contrast_feats = strong_contrast_feats.detach()

            # 监督损失（带标签平滑）
            focal_criterion = FocalLoss(alpha=0.8, gamma=2)# 新增损失函数
            loss_seg = F.cross_entropy(student_seg_out[:labeled_bs],label_batch[:labeled_bs], label_smoothing=0.1)
            outputs_soft = F.softmax(student_seg_out, dim=1)
            loss_seg_dice = losses.dice_loss(outputs_soft[:labeled_bs, 1, :, :, :],label_batch[:labeled_bs] == 1)
            loss_boundary = BoundaryLoss()(outputs_soft[:labeled_bs, 1], (label_batch[:labeled_bs] == 1).float())# 新增损失函数
            loss_focal = focal_criterion(student_seg_out[:labeled_bs], label_batch[:labeled_bs])# 新增损失函数
            supervised_loss = 0.3*(loss_seg + loss_seg_dice) + 0.4*loss_boundary + 0.3*loss_focal

            # 一致性损失（带动态mask）
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            consistency_dist = consistency_criterion(
                student_seg_out[labeled_bs:],
                label_smoothing(probs, factor=0.1)  # 教师标签平滑
            )
            # print(f"mask维度: {mask.shape}")
            weighted_loss = consistency_dist * mask  # 逐样本加权
            masked_consistency = weighted_loss.view(weighted_loss.shape[0], -1).mean(dim=1)
            consistency_loss = consistency_weight * torch.mean(weighted_loss)
            # ================= 新增：对比学习损失 =================
            _, _, weak_spatial_feats = student_model(weak_volume_batch, return_encoder_feats=True)
            _, _, strong_spatial_feats = student_model(strong_volume_batch, return_encoder_feats=True)

            contrast_loss = 0
            if contrast_enabled:
                for i in range(volume_batch.size(0)):
                    anchor_feat = weak_spatial_feats[i].unsqueeze(0)  # [1, C, D, H, W]
                    positive_feat = strong_spatial_feats[i].unsqueeze(0)  # [1, C, D, H, W]
                    if i < labeled_bs:
                        label_map = label_batch[i].unsqueeze(0).unsqueeze(1)
                        prob_map = None
                    else:
                        pseudo_label = torch.argmax(probs[i - labeled_bs], dim=0).unsqueeze(0).unsqueeze(1)
                        prob_map = max_probs[i - labeled_bs].unsqueeze(0).unsqueeze(1)
                        label_map = pseudo_label

                    # 新的 RCPS 对比损失
                    contrast_loss += student_model.contrast_learner.rcps_voxel_contrast(
                        anchor_feat,
                        positive_feat,
                        pseudo_labels=label_map,
                        prob_map=prob_map,
                        temperature=args.contrast_temp,
                        topk_neg=32  # 可配置
                    )
                contrast_loss = contrast_loss / volume_batch.size(0)
                contrast_weight = args.contrast_weight * min(1.0, (iter_num - args.contrast_start_iter) / 2000)
                weighted_contrast_loss = contrast_weight * contrast_loss
            else:
                weighted_contrast_loss = 0

            # 学生反向传播（带梯度裁剪）
            student_loss = supervised_loss + consistency_loss + weighted_contrast_loss
            student_optimizer.zero_grad()

            # 保留计算图供元学习
            with torch.enable_grad():
                student_loss.backward(retain_graph=True)
            meta_controller.update_weights(masked_consistency)  # 关键修改点
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip)  # 新增梯度裁剪
            student_optimizer.step()

            # ========== 阶段3：元学习教师更新 ==========
            # 生成元伪标签（带停止梯度）
            with torch.no_grad():
                meta_labels = torch.softmax(student_seg_out.detach(), dim=1)

            # 教师前向
            teacher_outputs = teacher_model(volume_batch, return_contrast_feats=False)

            # 教师损失计算
            teacher_supervised_loss = F.cross_entropy(
                teacher_outputs[:labeled_bs],
                label_batch[:labeled_bs].long(),  # 确保标签为Long类型
                label_smoothing=0.1,  # 内置标签平滑
                reduction='mean'
            )
            teacher_consistency_loss = losses.softmax_kl_loss(
                teacher_outputs[labeled_bs:],
                meta_labels[labeled_bs:]
            ).mean()

            # 动态权重调整
            teacher_weight = mpl_controller.get_teacher_weight()
            teacher_loss = teacher_supervised_loss + teacher_weight * teacher_consistency_loss

            # 教师反向传播（带元梯度）
            teacher_optimizer.zero_grad()
            teacher_loss.backward(retain_graph=True)

            # 在教师反向传播时：
            teacher_params = list(teacher_model.parameters())
            student_params = list(student_model.parameters())
            # 计算并应用元梯度
            meta_grads = mpl_controller.compute_meta_grad(
                teacher_loss=teacher_consistency_loss,  # 确保是标量损失
                student_params=list(student_model.parameters())  # 传递学生参数
            )
            for t_param, meta_g in zip(teacher_model.parameters(), meta_grads):
                if meta_g is not None:
                    t_param.grad += meta_g.to(t_param.device)

            torch.nn.utils.clip_grad_norm_(teacher_model.parameters(), args.grad_clip)  # 教师梯度裁剪
            teacher_optimizer.step()

            # ========== 阶段4：双向参数同步 ==========
            # 学生->教师软更新
            alpha_teacher = args.teacher_alpha
            with torch.no_grad():
                for t_param, s_param in zip(teacher_model.parameters(), student_model.parameters()):
                    t_param.data.mul_(alpha_teacher).add_(s_param.data, alpha=1 - alpha_teacher)

            # 教师->学生EMA同步
            update_ema_variables(teacher_model, student_model, alpha=0.999, global_step=iter_num)

            iter_num = iter_num + 1
            writer.add_scalar('uncertainty/mask_per', torch.sum(mask) / mask.numel(), iter_num)
            writer.add_scalar('uncertainty/threshold', threshold, iter_num)
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', student_loss, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('train/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('train/consistency_dist', torch.mean(consistency_dist), iter_num)
            # 记录对比学习损失
            if contrast_enabled:
                writer.add_scalar('loss/contrast_loss', contrast_loss, iter_num)
                writer.add_scalar('loss/weighted_contrast_loss', weighted_contrast_loss, iter_num)
            # logging.info('iteration %d : loss : %f cons_dist: %f, loss_weight: %f' %
            #              (iter_num, student_loss.item(), consistency_dist.item(), consistency_weight))
            logging.info('iteration %d : loss : %f  loss_weight: %f' %
                         (iter_num, student_loss.item(),  consistency_weight))

            ## change lr
            def adjust_learning_rate(optimizer, iteration, max_iter, base_lr):
                """余弦退火调度"""
                lr = base_lr * (math.cos(math.pi * iteration / max_iter) + 1) / 2
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            if iter_num % 2500 == 0:
                # 原线性调整改为：
                adjust_learning_rate(student_optimizer, iter_num, max_iterations, base_lr)
                adjust_learning_rate(teacher_optimizer, iter_num, max_iterations, base_lr * 0.1)
                # 动态调整对比学习参数
                if contrast_enabled:
                    epoch_ratio = iter_num / max_iterations
                    new_threshold = 0.25 + 0.15 * epoch_ratio
                    student_model.contrast_learner.edge_threshold = new_threshold
                    student_model.contrast_learner.loss_weights[0] = 1.0 - 0.3 * epoch_ratio
                    student_model.contrast_learner.loss_weights[1] = 0.7 + 0.3 * epoch_ratio
                    student_model.contrast_learner.topk_neg = int(24 + 8 * epoch_ratio)  # top-K动态调整
                    logging.info(
                        f"调整对比学习参数: edge_threshold={new_threshold:.3f}, weights={student_model.contrast_learner.loss_weights}")

            if iter_num % 1000 == 0:
                torch.save({
                    'student': student_model.state_dict(),
                    'teacher': teacher_model.state_dict(),
                    'contrast_learner': student_model.contrast_learner.state_dict(),  # 保存对比学习模块
                    'iter_num': iter_num
                }, os.path.join(snapshot_path, f'iter_{iter_num}.pth'))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break

    # 最终保存（保存所有模型）
    torch.save({
        'student': student_model.state_dict(),
        'teacher': teacher_model.state_dict(),
        'contrast_learner': student_model.contrast_learner.state_dict(),
    }, os.path.join(snapshot_path, f'iter_{max_iterations}.pth'))
    writer.close()