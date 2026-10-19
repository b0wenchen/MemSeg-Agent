"""Ported from our original preprocessing script; numerical operations preserved."""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
from glob import glob
from typing import Tuple, List
from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure
from scipy.ndimage import binary_fill_holes, binary_closing, generate_binary_structure
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from tqdm import tqdm


# =========================
# Utilities (沿用你的风格)
# =========================
def rotate_hwT_clockwise90(arr_hwT: np.ndarray) -> np.ndarray:
    """
    arr_hwT: (H,W,T)
    顺时针 90°： (H,W) -> (W,H)
    """
    if arr_hwT.ndim != 3:
        raise ValueError(f"Expect (H,W,T), got {arr_hwT.shape}")
    return np.rot90(arr_hwT, k=1, axes=(0, 1)).copy()


def contour_to_solid_multiclass(L_hwT: np.ndarray,
                                class_ids=(1, 3),
                                closing_iter: int = 2) -> np.ndarray:
    """
    把“轮廓线”label 变成“实心区域”label。
    L_hwT: (H,W,T) int16，值域包含 0/1/3（你的 CardiacUDA_test 映射后）
    - 对每一帧、每一类：closing -> fill_holes
    - 输出仍为 (H,W,T) int16，多类互斥：后写的类会覆盖前写的类（通常不会重叠）
    """
    H, W, T = L_hwT.shape
    out = np.zeros_like(L_hwT, dtype=np.int16)

    # 2D 结构元：8-connectivity
    struct = generate_binary_structure(rank=2, connectivity=2)

    for t in range(T):
        frame = L_hwT[:, :, t]
        filled_frame = np.zeros((H, W), dtype=np.int16)

        for cid in class_ids:
            bin_line = (frame == cid)
            if bin_line.sum() == 0:
                continue

            # closing: 让轮廓更闭合（处理断裂）
            bin_closed = binary_closing(bin_line, structure=struct, iterations=closing_iter)

            # fill holes: 填充内部
            bin_filled = binary_fill_holes(bin_closed)

            filled_frame[bin_filled] = cid

        out[:, :, t] = filled_frame

    return out
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def hist_cut_top(img_arr: np.ndarray, top_percent: float) -> np.ndarray:
    """截断最亮 top_percent%（例如 0.5 表示截断最亮 0.5%）"""
    if top_percent <= 0:
        return img_arr
    hi = float(np.percentile(img_arr, 100.0 - top_percent))
    out = img_arr.copy()
    out[out > hi] = hi
    return out

def compute_xy_center_from_label_zyx(label_zyx: np.ndarray, fallback_xy: Tuple[int, int]) -> Tuple[int, int]:
    """
    label_zyx: (Z,Y,X)
    返回 (cy,cx) in (Y,X)
    """
    fg = (label_zyx > 0)
    if fg.sum() < 10:
        return fallback_xy
    ys = np.where(fg)[1]
    xs = np.where(fg)[2]
    cy = int(np.rint(ys.mean()))
    cx = int(np.rint(xs.mean()))
    return (cy, cx)

def image_crop_yxz(ori_vol_yxz: np.ndarray, crop_hw: Tuple[int, int], center_yx: Tuple[int, int], padval=0.0) -> np.ndarray:
    """
    ori_vol_yxz: (Y,X,Z)
    return: (cropH,cropW,Z)
    """
    crop_h, crop_w = int(crop_hw[0]), int(crop_hw[1])
    Y, X, Z = ori_vol_yxz.shape
    out = np.ones((crop_h, crop_w, Z), dtype=ori_vol_yxz.dtype) * padval

    cy, cx = int(center_yx[0]), int(center_yx[1])
    y0 = cy - crop_h // 2
    x0 = cx - crop_w // 2
    y1 = y0 + crop_h
    x1 = x0 + crop_w

    src_y0 = max(y0, 0)
    src_x0 = max(x0, 0)
    src_y1 = min(y1, Y)
    src_x1 = min(x1, X)

    dst_y0 = src_y0 - y0
    dst_x0 = src_x0 - x0
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    if src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1, :] = ori_vol_yxz[src_y0:src_y1, src_x0:src_x1, :]

    return out


# =========================
# CardiacUDA specifics
# =========================
def resize_video_hwT(vol_hwT: np.ndarray, out_hw: Tuple[int, int], is_label: bool) -> np.ndarray:
    """
    vol_hwT: (H,W,T)
    return: (H2,W2,T)
    """
    import cv2
    oh, ow = int(out_hw[0]), int(out_hw[1])
    H, W, T = vol_hwT.shape
    if (H, W) == (oh, ow):
        return vol_hwT
    interp = cv2.INTER_NEAREST if is_label else cv2.INTER_LINEAR
    out = np.zeros((oh, ow, T), dtype=vol_hwT.dtype)
    for t in range(T):
        out[:, :, t] = cv2.resize(vol_hwT[:, :, t], (ow, oh), interpolation=interp)
    return out

def cardiacuda_a4c_to_camus_labelmap_hwT(L_hwT: np.ndarray) -> np.ndarray:
    """
    CardiacUDA A4C: 0 BG, 1 LV, 2 LA, 3 RA, 4 RV
    CAMUS label space: 0 BG, 1 LV, 2 Myo, 3 LA
    map: LV->1, LA->3, others->0
    """
    L = L_hwT.astype(np.int16)
    out = np.zeros_like(L, dtype=np.int16)
    out[L == 1] = 1
    out[L == 2] = 3
    return out

def list_pairs_label_all_frame(root: str) -> List[Tuple[str, str, str]]:
    """
    return list of (base, img_path, lab_path) under label_all_frame
    """
    imgs = sorted(glob(os.path.join(root, "*_image.nii.gz")))
    pairs = []
    for img_p in imgs:
        base = os.path.basename(img_p).replace("_image.nii.gz", "")
        lab_p = os.path.join(root, base + "_label.nii.gz")
        if os.path.exists(lab_p):
            pairs.append((base, img_p, lab_p))
    return pairs


# =========================
# Main preprocessing
# =========================
def preprocess_cardiacuda_allframe_video(
    root_label_all_frame: str,
    out_folder: str,
    resize_hw: Tuple[int, int] = (328, 328),
    crop_hw: Tuple[int, int] = (256, 256),
    hist_cut_top_percent: float = 0.5,
    logging: bool = True,
):
    """
    仅处理 label_all_frame 中：
      - image/label 成对
      - label unique ⊆ {0,1,2,3,4} （过滤掉异常编码）
    输出视频级 3D nii：
      - image_{idx}.nii.gz  (Z=T, Y=256, X=256)
      - label_{idx}.nii.gz  (Z=T, Y=256, X=256) mapped to CAMUS labels
      - index_mapping.csv
    """
    ensure_dir(out_folder)
    pairs = list_pairs_label_all_frame(root_label_all_frame)
    if not pairs:
        raise RuntimeError(f"No paired *_image/_label found under: {root_label_all_frame}")

    allowed = set([0, 1, 2, 3, 4])

    map_csv = os.path.join(out_folder, "index_mapping.csv")
    bad_csv = os.path.join(out_folder, "skipped.csv")

    with open(map_csv, "w", newline="", encoding="utf-8") as f_map, \
         open(bad_csv, "w", newline="", encoding="utf-8") as f_bad:
        wmap = csv.writer(f_map)
        wbad = csv.writer(f_bad)

        wmap.writerow(["new_index", "base", "num_frames_T", "orig_shape_hwT", "src_img", "src_gt", "note"])
        wbad.writerow(["base", "src_img", "src_gt", "reason"])

        idx = 1
        for base, img_p, lab_p in tqdm(pairs, desc="label_all_frame videos"):
            try:
                I = np.asarray(nib.load(img_p).dataobj).astype(np.float32)  # (H,W,T)
                L = np.asarray(nib.load(lab_p).dataobj).astype(np.int16)
            except Exception as e:
                wbad.writerow([base, img_p, lab_p, f"read_error: {repr(e)}"])
                continue

            if I.ndim != 3 or L.ndim != 3 or I.shape != L.shape:
                wbad.writerow([base, img_p, lab_p, f"shape_mismatch: I={I.shape}, L={L.shape}"])
                continue

            u = set(np.unique(L).astype(int).tolist())
            if not u.issubset(allowed):
                wbad.writerow([base, img_p, lab_p, f"bad_label_code: unique={sorted(list(u))}"])
                continue

            H, W, T = I.shape
            orig_shape = f"{H}x{W}x{T}"

            # map label -> CAMUS space
            L_camus = cardiacuda_a4c_to_camus_labelmap_hwT(L)
            # (1) rotate CW 90° for both image and label
            I = rotate_hwT_clockwise90(I)
            L_camus = rotate_hwT_clockwise90(L_camus)

            # (2) contour -> solid mask (LV=1, LA=3)
            L_camus = contour_to_solid_multiclass(L_camus, class_ids=(1, 3), closing_iter=2)

            # skip empty after mapping (rare)
            if np.sum(L_camus > 0) == 0:
                wbad.writerow([base, img_p, lab_p, "empty_mask_after_mapping"])
                continue

            # A) hist cut (whole video)
            I = hist_cut_top(I, hist_cut_top_percent).astype(np.float32)

            # B) resize
            I_rs = resize_video_hwT(I, resize_hw, is_label=False).astype(np.float32)
            L_rs = resize_video_hwT(L_camus, resize_hw, is_label=True).astype(np.int16)

            # C) crop center from union mask over time
            L_rs_zyx = L_rs.transpose(2, 0, 1)  # (T,H,W)
            fallback_xy = (L_rs_zyx.shape[1] // 2, L_rs_zyx.shape[2] // 2)
            ctr_yx = compute_xy_center_from_label_zyx(L_rs_zyx, fallback_xy)

            # D) crop all frames with same center
            # (H,W,T) == (Y,X,Z)
            crop_I_yxz = image_crop_yxz(I_rs, crop_hw=crop_hw, center_yx=ctr_yx, padval=float(I_rs.min())).astype(np.float32)
            crop_L_yxz = image_crop_yxz(L_rs, crop_hw=crop_hw, center_yx=ctr_yx, padval=0).astype(np.int16)

            # E) write (Z,Y,X) where Z=T
            crop_I_zyx = crop_I_yxz.transpose(2, 0, 1).astype(np.float32)  # (T,256,256)
            crop_L_zyx = crop_L_yxz.transpose(2, 0, 1).astype(np.int16)
            crop_L_zyx = np.clip(crop_L_zyx, 0, 3).astype(np.int16)

            out_img = sitk.GetImageFromArray(crop_I_zyx)
            out_lb  = sitk.GetImageFromArray(crop_L_zyx)

            # spacing: 统一像素 spacing
            out_img.SetSpacing((1.0, 1.0, 1.0))
            out_lb.SetSpacing((1.0, 1.0, 1.0))

            out_img_f = os.path.join(out_folder, f"image_{idx}.nii.gz")
            out_lb_f  = os.path.join(out_folder, f"label_{idx}.nii.gz")
            sitk.WriteImage(out_img, out_img_f, True)
            sitk.WriteImage(out_lb,  out_lb_f,  True)

            wmap.writerow([idx, base, T, orig_shape, img_p, lab_p,
                           "from label_all_frame; keep only unique⊆{0..4}; map LV->1 LA->3 others->0; output Z=T"])

            if logging:
                print(f"[Saved #{idx}] {out_img_f}")

            idx += 1

    print(f"\nDone. Output folder: {out_folder}")
    print(f"Index mapping saved to: {map_csv}")
    print(f"Skipped list saved to: {bad_csv}")
