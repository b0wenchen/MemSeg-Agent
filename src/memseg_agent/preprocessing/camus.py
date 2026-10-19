"""Ported from our original preprocessing script; numerical operations preserved."""
import os
import re
import csv
import numpy as np
import SimpleITK as sitk
from glob import glob

# =========================
# Utilities
# =========================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def is_half_sequence(path: str) -> bool:
    return "half_sequence" in os.path.basename(path).lower()

def is_gt_file(path: str) -> bool:
    base = os.path.basename(path).lower()
    return ("_gt" in base) or ("_seg" in base) or ("_mask" in base)

def parse_view_phase(path: str):
    """
    从文件名中解析 view (2CH/4CH) 和 phase (ED/ES)
    只用于 ED/ES 静态帧
    """
    base = os.path.basename(path).upper()
    view = None
    phase = None

    if "2CH" in base:
        view = "2CH"
    elif "4CH" in base:
        view = "4CH"

    # 注意：避免匹配 half_sequence（我们会在上层过滤）
    if "_ED" in base:
        phase = "ED"
    elif "_ES" in base:
        phase = "ES"

    return view, phase

def read_image(path: str) -> sitk.Image:
    return sitk.ReadImage(path)

def hist_cut_top(img_arr: np.ndarray, top_percent: float) -> np.ndarray:
    """
    截断最亮 top_percent%（例如 0.5 表示截断最亮 0.5%）
    """
    if top_percent <= 0:
        return img_arr
    hi = float(np.percentile(img_arr, 100.0 - top_percent))
    out = img_arr.copy()
    out[out > hi] = hi
    return out

def resample_2d_to_spacing(img: sitk.Image, new_spacing_xy=(0.5, 0.5),
                           interpolator=sitk.sitkLinear, logging=False) -> sitk.Image:
    """
    仅支持 2D 图像重采样到统一 spacing
    """
    if img.GetDimension() != 2:
        raise RuntimeError(f"resample_2d_to_spacing expects 2D image, got dim={img.GetDimension()}")

    old_spacing = img.GetSpacing()  # (sx, sy)
    old_size = img.GetSize()        # (W, H) in SITK index order

    new_spacing = (float(new_spacing_xy[0]), float(new_spacing_xy[1]))

    # 新 size = old_size * old_spacing / new_spacing
    new_size = [
        int(np.round(old_size[0] * old_spacing[0] / new_spacing[0])),
        int(np.round(old_size[1] * old_spacing[1] / new_spacing[1])),
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())

    if logging:
        print(f"  [Resample2D] spacing {old_spacing} -> {new_spacing}, size {old_size} -> {new_size}")

    return resampler.Execute(img)

def crop_2d_centered(arr_yx: np.ndarray, crop_hw=(256, 256), center_yx=None, pad_val=0.0) -> np.ndarray:
    """
    arr_yx: (H, W)
    crop_hw: (crop_h, crop_w)
    center_yx: (cy, cx). 若 None 则用图像中心
    """
    H, W = arr_yx.shape
    ch, cw = int(crop_hw[0]), int(crop_hw[1])

    if center_yx is None:
        cy, cx = H // 2, W // 2
    else:
        cy, cx = int(center_yx[0]), int(center_yx[1])

    # 目标框在原图上的范围
    y0 = cy - ch // 2
    x0 = cx - cw // 2
    y1 = y0 + ch
    x1 = x0 + cw

    out = np.ones((ch, cw), dtype=arr_yx.dtype) * pad_val

    # 计算与原图的交集
    src_y0 = max(y0, 0)
    src_x0 = max(x0, 0)
    src_y1 = min(y1, H)
    src_x1 = min(x1, W)

    dst_y0 = src_y0 - y0
    dst_x0 = src_x0 - x0
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    if src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = arr_yx[src_y0:src_y1, src_x0:src_x1]

    return out

def compute_center_from_mask(mask_yx: np.ndarray, fallback_yx):
    fg = mask_yx > 0
    if fg.sum() < 10:
        return fallback_yx
    ys, xs = np.where(fg)
    cy = int(np.round(ys.mean()))
    cx = int(np.round(xs.mean()))
    return (cy, cx)

def find_ed_es_pairs(patient_dir: str):
    """
    返回 list[(img_path, gt_path, view, phase)]
    只收集 2CH/4CH + ED/ES，忽略 half_sequence
    """
    nii_files = glob(os.path.join(patient_dir, "*.nii")) + glob(os.path.join(patient_dir, "*.nii.gz"))
    nii_files = [f for f in nii_files if not is_half_sequence(f)]

    buckets = {}  # (view, phase) -> {"img":..., "gt":...}

    for f in nii_files:
        view, phase = parse_view_phase(f)
        if view is None or phase is None:
            continue
        key = (view, phase)
        if key not in buckets:
            buckets[key] = {"img": None, "gt": None}

        if is_gt_file(f):
            buckets[key]["gt"] = f
        else:
            buckets[key]["img"] = f

    pairs = []
    for (view, phase), d in buckets.items():
        if d["img"] and d["gt"]:
            pairs.append((d["img"], d["gt"], view, phase))

    # 固定排序：2CH ED, 2CH ES, 4CH ED, 4CH ES
    order = {"2CH": 0, "4CH": 1}
    phase_order = {"ED": 0, "ES": 1}
    pairs.sort(key=lambda x: (order.get(x[2], 99), phase_order.get(x[3], 99)))
    return pairs

def write_as_pseudo3d(arr_yx: np.ndarray, ref_img_2d: sitk.Image) -> sitk.Image:
    """
    将 2D (H,W) 写成伪3D (Z=1,Y,X)，以兼容你按 shape[2] 切片的 Dataset
    SITK GetArrayFromImage/FromArray 采用 (Z,Y,X)
    """
    arr_zyx = arr_yx[None, ...]  # (1,H,W)
    out = sitk.GetImageFromArray(arr_zyx)

    # 方向/原点：保留2D的前提下，我们把它写成3D会缺一个维度的direction
    # 最稳妥是：只设置 spacing/origin 的前两维，第三维给 1.0 与 0.0
    sp = ref_img_2d.GetSpacing()  # (sx,sy)
    org = ref_img_2d.GetOrigin()  # (ox,oy)

    out.SetSpacing((float(sp[0]), float(sp[1]), 1.0))
    out.SetOrigin((float(org[0]), float(org[1]), 0.0))

    # direction：用单位矩阵（避免2D->3D的方向不兼容）
    out.SetDirection((1.0,0.0,0.0,
                      0.0,1.0,0.0,
                      0.0,0.0,1.0))
    return out

# =========================
# Main preprocessing
# =========================
def preprocess_camus_2d(
    camus_root: str,
    out_folder: str,
    crop_hw=(256, 256),
    hist_cut_top_percent=0.5,
    new_spacing_xy=(0.5, 0.5),
    logging=True
):
    ensure_dir(out_folder)

    patients = sorted([
        os.path.join(camus_root, d)
        for d in os.listdir(camus_root)
        if os.path.isdir(os.path.join(camus_root, d)) and re.match(r"patient\d+", d, flags=re.I)
    ])
    if not patients:
        raise RuntimeError(f"No patient folders found under: {camus_root}")

    map_csv = os.path.join(out_folder, "index_mapping.csv")
    with open(map_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["new_index", "patient_id", "view", "phase", "src_img", "src_gt"])

        idx = 1
        for pdir in patients:
            pid = os.path.basename(pdir)
            pairs = find_ed_es_pairs(pdir)

            if logging:
                print(f"\n=== {pid}: found {len(pairs)} view/phase pairs ===")

            for img_path, gt_path, view, phase in pairs:
                if logging:
                    print(f"--- Processing {pid} {view} {phase}")

                img2d = read_image(img_path)
                gt2d = read_image(gt_path)

                if img2d.GetDimension() != 2 or gt2d.GetDimension() != 2:
                    raise RuntimeError(
                        f"Expected 2D ED/ES for CAMUS. Got img dim={img2d.GetDimension()}, gt dim={gt2d.GetDimension()} "
                        f"for {img_path}"
                    )

                # 强制 gt geometry 继承 img（避免 header 异常）
                gt2d.SetSpacing(img2d.GetSpacing())
                gt2d.SetOrigin(img2d.GetOrigin())
                gt2d.SetDirection(img2d.GetDirection())

                # ---- A) histogram cut on image ----
                img_arr = sitk.GetArrayFromImage(img2d).astype(np.float32)  # (H,W)
                img_arr = hist_cut_top(img_arr, hist_cut_top_percent)
                img_his = sitk.GetImageFromArray(img_arr)
                img_his.CopyInformation(img2d)

                # ---- B) resample image and label to unified 2D spacing ----
                img_rs = resample_2d_to_spacing(img_his, new_spacing_xy, interpolator=sitk.sitkLinear, logging=logging)

                # label 用 reference image + nearest neighbor（最稳）
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(img_rs)
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                resampler.SetTransform(sitk.Transform())
                gt_rs = resampler.Execute(gt2d)

                # ---- C) crop to unified size using mask center ----
                img_rs_arr = sitk.GetArrayFromImage(img_rs).astype(np.float32)  # (H,W)
                gt_rs_arr = sitk.GetArrayFromImage(gt_rs).astype(np.int16)      # (H,W)

                fallback = (img_rs_arr.shape[0] // 2, img_rs_arr.shape[1] // 2)  # (cy,cx)
                ctr = compute_center_from_mask(gt_rs_arr, fallback)

                crop_img = crop_2d_centered(
                    img_rs_arr, crop_hw=crop_hw, center_yx=ctr, pad_val=float(img_rs_arr.min())
                ).astype(np.float32)

                crop_gt = crop_2d_centered(
                    gt_rs_arr, crop_hw=crop_hw, center_yx=ctr, pad_val=0
                ).astype(np.int16)

                # CAMUS 常见 label 0..3，安全 clip
                crop_gt = np.clip(crop_gt, 0, 3).astype(np.int16)

                # ---- D) save as pseudo-3D (Z=1,Y,X) ----
                out_img_obj = write_as_pseudo3d(crop_img, img_rs)
                out_gt_obj = write_as_pseudo3d(crop_gt, img_rs)

                out_img_f = os.path.join(out_folder, f"image_{idx}.nii.gz")
                out_gt_f = os.path.join(out_folder, f"label_{idx}.nii.gz")

                sitk.WriteImage(out_img_obj, out_img_f, True)
                sitk.WriteImage(out_gt_obj, out_gt_f, True)

                writer.writerow([idx, pid, view, phase, img_path, gt_path])

                if logging:
                    print(f"[Saved] {out_img_f}")
                    print(f"[Saved] {out_gt_f}")

                idx += 1

    print(f"\nDone. Output folder: {out_folder}")
    print(f"Index mapping saved to: {map_csv}")
