import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import argparse
import torch
from networks.vnet_cl import VNet
from test_util import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/zlj/workspace/tjy/MeTi-SSL/data/2018LA_Seg_Training Set/',
                    help='Dataset root path')
# parser.add_argument('--root_path', type=str, default='/root/autodl-tmp/MeTi/data/2018LA_Seg_Training Set/2018LA_Seg_Training Set/',
#                     help='Dataset root path')
parser.add_argument('--model', type=str, default='train_cl', help='Model name')
parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
##### MPL MOD START 新增测试模式参数
parser.add_argument('--test_mode', type=str, default='student',
                    choices=['student', 'teacher', 'contrast_learner'], help='Which model to test')
##### MPL MOD END
FLAGS = parser.parse_args()

snapshot_path = "../model/" + FLAGS.model + "/"
test_save_path = "../model/prediction/{}_{}_post/".format(FLAGS.model, FLAGS.test_mode)  ##### MPL MOD
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

num_classes = 2

with open(FLAGS.root_path + '/../test.list', 'r') as f:
    image_list = f.readlines()
image_list = [FLAGS.root_path + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]


##### MPL MOD START 修改模型加载方式
def test_calculate_metric(epoch_num):
    # 初始化EMA模型（测试时推荐使用）
    net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=False, mc_dropout=False).cuda()

    # 加载检查点
    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(epoch_num) + '.pth')
    checkpoint = torch.load(save_mode_path)

    # 根据模式选择加载的模型参数
    if FLAGS.test_mode == 'student':
        net.load_state_dict(checkpoint['student'])
    elif FLAGS.test_mode == 'teacher':
        net.load_state_dict(checkpoint['teacher'])
    else:  # ema
        net.load_state_dict(checkpoint['contrast_learner'])

    print("Loaded {} model weights from {}".format(FLAGS.test_mode, save_mode_path))
    net.eval()

    # 保持原有测试参数
    avg_metric = test_all_case(
        net, image_list, num_classes=num_classes,
        patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
        save_result=True, test_save_path=test_save_path
    )
    return avg_metric


##### MPL MOD END

if __name__ == '__main__':
    # metric = test_calculate_metric(15000)
    # print(f"Results: {metric}\n")
    metric = test_calculate_metric(14000)
    print(f"Results: {metric}\n")
    # metric = test_calculate_metric(13000)
    # print(f"Results: {metric}\n")
    # metric = test_calculate_metric(12000)
    # print(f"Results: {metric}\n")
    # metric = test_calculate_metric(11000)
    # print(f"Results: {metric}\n")
    # metric = test_calculate_metric(10000)
    # print(f"Results: {metric}\n")
    # metric = test_calculate_metric(9000)
    # print(f"Results: {metric}\n")
    # metric = test_calculate_metric(8000)
    # print(f"Results: {metric}\n")
    # metric = test_calculate_metric(7000)
    # print(f"Results: {metric}\n")
    # metric = test_calculate_metric(6000)
    # print(f"Results: {metric}\n")
    # metric = test_calculate_metric(5000)
    # print(f"Results: {metric}\n")
    # metric = test_calculate_metric(4000)
    # print(f"Results: {metric}\n")
    # metric = test_calculate_metric(3000)
    # print(f"Results: {metric}\n")
    # metric = test_calculate_metric(2000)
    # print(f"Results: {metric}\n")
    # metric = test_calculate_metric(1000)
    # print(f"Results: {metric}\n")
