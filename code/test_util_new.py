import os

import h5py
import math
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import plotly.graph_objects as go
from skimage import measure
from scipy.ndimage import zoom
import os

def plotly_3d_surface_smooth(binary_mask, save_path, color='red', title='', scale_factor=4, mc_step_size=1):

    mask_highres = zoom(binary_mask, zoom=scale_factor, order=3)

    verts, faces, normals, values = measure.marching_cubes(mask_highres, level=0.5, step_size=mc_step_size)

    mesh = go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color, opacity=1.0, name=title
    )

    fig = go.Figure(data=[mesh])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data',
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    # ✅ 初始可读视角
    fig.update_layout(scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)))

    # ✅ 自由轨迹球旋转
    fig.update_scenes(dragmode="orbit")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.write_html(save_path)
    print(f"Saved: {save_path}")


def save_all_overlay_slices(image, label, prediction, save_dir, id):
    os.makedirs(save_dir, exist_ok=True)
    num_slices = image.shape[-1]
    for i in range(num_slices):
        img_slice = image[..., i]
        label_slice = label[..., i]
        pred_slice = prediction[..., i]

        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(img_slice, cmap='gray', alpha=0.7)
        # 标签红色轮廓线
        ax.contour(label_slice, levels=[0.5], colors='red', linewidths=2)
        # 预测黄色轮廓线
        ax.contour(pred_slice, levels=[0.5], colors='yellow', linewidths=2)
        ax.axis('off')
        plt.tight_layout()
        out_path = os.path.join(save_dir, f"{id}_slice_{i:03d}_overlay.png")
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)


def test_all_case(net, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=True,
                  test_save_path=None, preproc_fn=None):
    total_metric = 0.0
    for image_path in tqdm(image_list):
        id = image_path.split('/')[-1]
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)

        if np.sum(prediction) == 0:
            single_metric = (0, 0, 0, 0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])
            print(single_metric)
        total_metric += np.asarray(single_metric)

        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path + id + "_pred.nii.gz")
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path + id + "_img.nii.gz")
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + id + "_gt.nii.gz")
        # overlay_dir = os.path.join(test_save_path, id + "_overlay_slices")
        # save_all_overlay_slices(image, label, prediction, overlay_dir, id)
        # 替换原本的散点图函数
        plotly_3d_surface_smooth(label, os.path.join(test_save_path, f"{id}_label3d.html"), color='red', title='Label')
        plotly_3d_surface_smooth(prediction, os.path.join(test_save_path, f"{id}_prediction3d.html"), color='yellow',
                                 title='Prediction')

    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))

    return avg_metric


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant',
                       constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                y1 = net(test_patch)
                y = F.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + y
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + 1
    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    return label_map, score_map


def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num - 1)
    for i in range(1, num):
        prediction_tmp = (prediction == i)
        label_tmp = (label == i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd
