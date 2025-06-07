"""
6000次：[0.88135327  0.79014253 12.25124604  3.49887661]
5000次：[0.88263358  0.79238839 11.50504356  3.16043276]
4000次：[0.8815926   0.79074633 11.94703053  3.27772464]
3000次：[0.88297314  0.79272962 12.11338875  3.2815329 ]
2000次：[0.87139157  0.7753485  12.70827868  3.79566673]
1000次：[0.8469684   0.73961901 14.73560037  3.8248198 ]
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.vnet import VNet
from dataloaders import utils
from utils import ramps, losses
from dataloaders.la_heart import LAHeart, RandomCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler

from utils.contrastive_losses import voxel_contrastive_loss, patch_contrastive_loss
from utils.region_mask import get_region_masks

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/zlj/workspace/tjy/MeTi-SSL/data/2018LA_Seg_Training Set/', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='CL1', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--lambda_voxel', type=float, default=0.05, help='voxel contrast loss weight')
parser.add_argument('--lambda_patch', type=float, default=0.05, help='patch contrast loss weight')
parser.add_argument('--lambda_kd', type=float, default=0.01, help='feature distill loss weight')
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
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (112, 112, 80)

def get_current_consistency_weight(epoch):
    return 0.1 * ramps.sigmoid_rampup(epoch, 40.0)

def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def kd_loss(student_feats, teacher_feats):
    # 多尺度特征蒸馏（L2）
    loss = 0.0
    for s, t in zip(student_feats, teacher_feats):
        loss += F.mse_loss(s, t.detach())
    return loss / len(student_feats)

if __name__ == "__main__":
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(ema=False):
        net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))
    labeled_idxs = list(range(16))
    unlabeled_idxs = list(range(16, 80))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,worker_init_fn=worker_init_fn)

    model.train()
    ema_model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    model.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            noise = torch.clamp(torch.randn_like(volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = volume_batch + noise
            # ----------- 网络输出，多尺度特征 ------------
            outputs, student_feats = model(volume_batch, return_features=True)
            with torch.no_grad():
                ema_output, teacher_feats = ema_model(ema_inputs, return_features=True)
            outputs_soft = F.softmax(outputs, dim=1)
            T = 8
            volume_batch_r = volume_batch.repeat(2, 1, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 2
            preds = torch.zeros([stride * T, 2, 112, 112, 80]).cuda()
            for i in range(T // 2):
                ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
                with torch.no_grad():
                    preds[2 * stride * i:2 * stride * (i + 1)] = ema_model(ema_inputs)
            preds = F.softmax(preds, dim=1)
            preds = preds.reshape(T, stride, 2, 112, 112, 80)
            preds = torch.mean(preds, dim=0)
            uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)
            # ----------- 区域掩码 ------------
            threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(iter_num, max_iterations)) * np.log(2)
            edge_mask, core_mask = get_region_masks(outputs_soft, uncertainty, threshold=threshold, edge_kernel=3)
            # 下采样 edge_mask 和 core_mask 到 student_feats[-2] 的空间尺寸
            edge_mask_ds = F.interpolate(edge_mask.float(), size=student_feats[-2].shape[2:], mode='nearest')
            core_mask_ds = F.interpolate(core_mask.float(), size=student_feats[-2].shape[2:], mode='nearest')
            # ----------- 损失计算 ------------
            loss_seg = F.cross_entropy(outputs[:labeled_bs], label_batch[:labeled_bs])
            loss_seg_dice = losses.dice_loss(outputs_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
            # 一致性损失
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            consistency_dist = losses.softmax_mse_loss(outputs, ema_output)
            mask = (uncertainty < threshold).float()
            consistency_dist = torch.sum(mask * consistency_dist) / (2 * torch.sum(mask) + 1e-16)
            consistency_loss = consistency_weight * consistency_dist
            # 边缘体素对比
            loss_voxel = voxel_contrastive_loss(student_feats[-2], edge_mask_ds)
            # 核心补丁对比
            loss_patch = patch_contrastive_loss(student_feats[-2], core_mask_ds)
            # 多尺度蒸馏
            loss_kd = kd_loss(student_feats, teacher_feats)
            # 总损失
            loss = 0.5 * (loss_seg + loss_seg_dice) + consistency_loss + \
                   args.lambda_voxel * loss_voxel + args.lambda_patch * loss_patch + args.lambda_kd * loss_kd
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, 0.99, iter_num)

            iter_num = iter_num + 1
            writer.add_scalar('uncertainty/mean', uncertainty[0,0].mean(), iter_num)
            writer.add_scalar('uncertainty/max', uncertainty[0,0].max(), iter_num)
            writer.add_scalar('uncertainty/min', uncertainty[0,0].min(), iter_num)
            writer.add_scalar('uncertainty/mask_per', torch.sum(mask)/mask.numel(), iter_num)
            writer.add_scalar('uncertainty/threshold', threshold, iter_num)
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('train/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('train/consistency_dist', consistency_dist, iter_num)

            logging.info('iteration %d : loss : %f cons_dist: %f, loss_weight: %f' %
                         (iter_num, loss.item(), consistency_dist.item(), consistency_weight))
            if iter_num % 50 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                # image = outputs_soft[0, 3:4, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                image = torch.max(outputs_soft[0, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label', grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].permute(2, 0, 1)
                grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
                writer.add_image('train/Groundtruth_label', grid_image, iter_num)

                image = uncertainty[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/uncertainty', grid_image, iter_num)

                mask2 = (uncertainty > threshold).float()
                image = mask2[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/mask', grid_image, iter_num)
                #####
                image = volume_batch[-1, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('unlabel/Image', grid_image, iter_num)

                # image = outputs_soft[-1, 3:4, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                image = torch.max(outputs_soft[-1, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('unlabel/Predicted_label', grid_image, iter_num)

                image = label_batch[-1, :, :, 20:61:10].permute(2, 0, 1)
                grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
                writer.add_image('unlabel/Groundtruth_label', grid_image, iter_num)

            ## change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations)+'.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
