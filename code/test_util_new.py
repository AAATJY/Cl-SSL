import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import imageio.v2 as imageio


def _normalize_to_uint8(img2d: np.ndarray) -> np.ndarray:
    mn, mx = float(img2d.min()), float(img2d.max())
    if mx > mn:
        arr = (img2d - mn) / (mx - mn)
    else:
        arr = np.zeros_like(img2d, dtype=np.float32)
    return (arr * 255.0).clip(0, 255).astype(np.uint8)


def _save_prediction_pngs(image: np.ndarray,
                          prediction: np.ndarray,
                          label: np.ndarray,
                          out_dir: str,
                          base_id: str,
                          axis: int = 2,
                          overlay: bool = True):
    """
    将3D图像的每个切片保存为PNG；同时可保存分割叠加图
    image: [D,H,W] 或 [W,H,D]（保持与调用处一致，这里按 test 单例传入）
    prediction: 同尺寸整数标签（0/1/…）
    label: 同尺寸整数标签（0/1/…）
    axis: 0=x, 1=y, 2=z（默认轴向切片）
    """
    os.makedirs(out_dir, exist_ok=True)
    assert image.shape == prediction.shape == label.shape, "image/pred/label 形状需一致"

    num_slices = image.shape[axis]
    for idx in range(num_slices):
        if axis == 0:
            img2d = image[idx, :, :]
            pred2d = prediction[idx, :, :]
            gt2d = label[idx, :, :]
        elif axis == 1:
            img2d = image[:, idx, :]
            pred2d = prediction[:, idx, :]
            gt2d = label[:, idx, :]
        else:
            img2d = image[:, :, idx]
            pred2d = prediction[:, :, idx]
            gt2d = label[:, :, idx]

        img_u8 = _normalize_to_uint8(img2d)
        pred_u8 = (pred2d.astype(np.uint8) * 255)
        gt_u8 = (gt2d.astype(np.uint8) * 255)

        imageio.imwrite(os.path.join(out_dir, f"{base_id}_img_{idx:03d}.png"), img_u8)
        imageio.imwrite(os.path.join(out_dir, f"{base_id}_pred_{idx:03d}.png"), pred_u8)
        imageio.imwrite(os.path.join(out_dir, f"{base_id}_gt_{idx:03d}.png"), gt_u8)

        if overlay:
            # 将灰度转为RGB，并将预测掩膜以红色叠加
            rgb = np.stack([img_u8, img_u8, img_u8], axis=-1).astype(np.float32)
            mask = pred2d.astype(bool)
            mask3 = np.stack([mask, mask, mask], axis=-1)
            color = np.array([255.0, 0.0, 0.0], dtype=np.float32)  # 红色
            alpha = 0.4
            overlay_img = rgb.copy()
            overlay_img[mask3] = (alpha * color + (1 - alpha) * overlay_img[mask3])
            overlay_img = overlay_img.clip(0, 255).astype(np.uint8)
            imageio.imwrite(os.path.join(out_dir, f"{base_id}_overlay_{idx:03d}.png"), overlay_img)


def test_all_case(net,
                  image_list,
                  num_classes,
                  patch_size=(112, 112, 80),
                  stride_xy=18,
                  stride_z=4,
                  save_result=True,
                  test_save_path=None,
                  preproc_fn=None,
                  save_png=False,
                  png_save_path=None,
                  overlay_png=False,
                  slice_axis='z'):
    """
    新增导出PNG切片图像功能：
    - save_png=True 时，按 slice_axis 导出整卷每张切片的图像/预测/GT及叠加图（可选）
    - slice_axis: 'x'/'y'/'z' -> 0/1/2
    """
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis = axis_map.get(slice_axis, 2)

    total_metric = 0.0
    for image_path in tqdm(image_list):
        id = image_path.split('/')[-1].replace('.h5', '')
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
        total_metric += np.asarray(single_metric)

        if save_result:
            os.makedirs(test_save_path, exist_ok=True)
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), os.path.join(test_save_path, id + "_pred.nii.gz"))
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), os.path.join(test_save_path, id + "_img.nii.gz"))
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), os.path.join(test_save_path, id + "_gt.nii.gz"))

        if save_png:
            case_png_dir = os.path.join(png_save_path if png_save_path is not None else test_save_path, id)
            _save_prediction_pngs(image=image,
                                  prediction=prediction,
                                  label=label,
                                  out_dir=case_png_dir,
                                  base_id=id,
                                  axis=axis,
                                  overlay=overlay_png)

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