"""Ported from our original preprocessing script; numerical operations preserved."""
import os
import re
import csv
import numpy as np
import SimpleITK as sitk
import re
# =========================
# Helper functions (沿用你的风格)
# =========================
def resample_by_res(mov_img_obj, new_spacing, interpolator=sitk.sitkLinear, logging=True):
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetOutputDirection(mov_img_obj.GetDirection())
    resample.SetOutputOrigin(mov_img_obj.GetOrigin())
    mov_spacing = mov_img_obj.GetSpacing()

    resample.SetOutputSpacing(new_spacing)
    RES_COE = np.array(mov_spacing) * 1.0 / np.array(new_spacing)
    new_size = np.array(mov_img_obj.GetSize()) * RES_COE
    resample.SetSize([int(sz + 1) for sz in new_size])

    if logging:
        print(f"Spacing: {mov_spacing} -> {new_spacing}")
        print(f"Size: {mov_img_obj.GetSize()} -> {new_size}")

    return resample.Execute(mov_img_obj)

def resample_lb_by_res(mov_lb_obj, new_spacing, interpolator=sitk.sitkLinear, ref_img=None, logging=True):
    src_mat = sitk.GetArrayFromImage(mov_lb_obj)  # (z,y,x)
    lbvs = np.unique(src_mat)
    if logging:
        print("Label values:", lbvs)

    out_vol = None
    _tar_curr_obj = None
    for idx, lbv in enumerate(lbvs):
        _src_curr_mat = np.float32(src_mat == lbv)
        _src_curr_obj = sitk.GetImageFromArray(_src_curr_mat)
        _src_curr_obj.CopyInformation(mov_lb_obj)
        _tar_curr_obj = resample_by_res(_src_curr_obj, new_spacing, interpolator, logging=False)
        _tar_curr_mat = np.rint(sitk.GetArrayFromImage(_tar_curr_obj)) * lbv

        if out_vol is None:
            out_vol = _tar_curr_mat
        else:
            out_vol[_tar_curr_mat == lbv] = lbv

    out_obj = sitk.GetImageFromArray(out_vol.astype(np.int16))
    out_obj.SetSpacing(_tar_curr_obj.GetSpacing())
    if ref_img is not None:
        out_obj.CopyInformation(ref_img)
    return out_obj

def image_crop(ori_vol, crop_size, referece_ctr_idx, padval=0., only_2d=True):
    _expand_cropsize = [x + 1 for x in crop_size]
    if only_2d:
        assert len(crop_size) == 2
        assert len(referece_ctr_idx) == 2
        _expand_cropsize.append(ori_vol.shape[-1])

    image_patch = np.ones(tuple(_expand_cropsize)) * padval
    half_size = tuple([int(x * 1.0 / 2) for x in _expand_cropsize])
    _min_idx = [0, 0, 0]
    _max_idx = list(ori_vol.shape)
    _bias_start = [0, 0, 0]
    _bias_end = [0, 0, 0]

    for dim, hsize in enumerate(half_size):
        if dim == 2 and only_2d:
            break
        _bias_start[dim] = np.min([hsize, referece_ctr_idx[dim]])
        _bias_end[dim] = np.min([hsize, ori_vol.shape[dim] - referece_ctr_idx[dim]])
        _min_idx[dim] = referece_ctr_idx[dim] - _bias_start[dim]
        _max_idx[dim] = referece_ctr_idx[dim] + _bias_end[dim]

    if only_2d:
        image_patch[
            half_size[0] - _bias_start[0]: half_size[0] + _bias_end[0],
            half_size[1] - _bias_start[1]: half_size[1] + _bias_end[1],
            ...
        ] = ori_vol[
            referece_ctr_idx[0] - _bias_start[0]: referece_ctr_idx[0] + _bias_end[0],
            referece_ctr_idx[1] - _bias_start[1]: referece_ctr_idx[1] + _bias_end[1],
            ...
        ]
        image_patch = image_patch[0:crop_size[0], 0:crop_size[1], :]
    else:
        image_patch[
            half_size[0] - _bias_start[0]: half_size[0] + _bias_end[0],
            half_size[1] - _bias_start[1]: half_size[1] + _bias_end[1],
            half_size[2] - _bias_start[2]: half_size[2] + _bias_end[2]
        ] = ori_vol[
            referece_ctr_idx[0] - _bias_start[0]: referece_ctr_idx[0] + _bias_end[0],
            referece_ctr_idx[1] - _bias_start[1]: referece_ctr_idx[1] + _bias_end[1],
            referece_ctr_idx[2] - _bias_start[2]: referece_ctr_idx[2] + _bias_end[2]
        ]
        image_patch = image_patch[0:crop_size[0], 0:crop_size[1], 0:crop_size[2]]
    return image_patch

def copy_spacing_ori(src, dst):
    dst.SetSpacing(src.GetSpacing())
    dst.SetOrigin(src.GetOrigin())
    dst.SetDirection(src.GetDirection())
    return dst

def find_phase_files(patient_dir, phase):  # phase in {"ED","ES"}
    # 支持 .nii 或 .nii.gz
    # ACDC 命名通常是 patientXXX_ED.nii.gz / patientXXX_ED_gt.nii.gz
    pid = os.path.basename(patient_dir)
    candidates_img = [
        os.path.join(patient_dir, f"{pid}_{phase}.nii.gz"),
        os.path.join(patient_dir, f"{pid}_{phase}.nii")
    ]
    candidates_gt = [
        os.path.join(patient_dir, f"{pid}_{phase}_gt.nii.gz"),
        os.path.join(patient_dir, f"{pid}_{phase}_gt.nii")
    ]
    img = next((p for p in candidates_img if os.path.exists(p)), None)
    gt  = next((p for p in candidates_gt if os.path.exists(p)), None)
    return img, gt

def compute_xy_center_from_label(label_zyx, fallback_xy):
    """
    label_zyx: np array, shape (Z,Y,X) from sitk.GetArrayFromImage
    返回 (cy, cx) 以适配你的 image_crop 中 transpose(1,2,0) 的 (Y,X,Z) 逻辑
    """
    fg = (label_zyx > 0)
    if fg.sum() < 10:
        return fallback_xy

    # 只关心 y,x 的中心（对 z 取 union）
    # fg indices: (z,y,x)
    ys = np.where(fg)[1]
    xs = np.where(fg)[2]
    cy = int(np.rint(ys.mean()))
    cx = int(np.rint(xs.mean()))
    return (cy, cx)

def frame_name(pid, frame_idx):
    return f"{pid}_frame{int(frame_idx):02d}.nii.gz"

def frame_gt_name(pid, frame_idx):
    return f"{pid}_frame{int(frame_idx):02d}_gt.nii.gz"
def read_info_cfg(cfg_path):
    info = {}
    with open(cfg_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # 支持 "ED = 1" 和 "ED: 1"
            if ":" in line:
                k, v = line.split(":", 1)
            elif "=" in line:
                k, v = line.split("=", 1)
            else:
                continue

            k = k.strip()
            v = v.strip()

            # 统一 key（可选，但更稳）
            k = k.upper()

            # 去掉可能的多余字符
            v = re.sub(r"\s+", "", v)

            info[k] = v

    return info

# =========================
# Main preprocessing
# =========================
def preprocess_acdc(
    acdc_root,
    out_folder,
    crop_hw=(256, 256),
    hist_cut_top=0.5,              # 截断 top 0.5%
    new_spacing=(1.25, 1.25, 7.70),
    logging=True
):
    os.makedirs(out_folder, exist_ok=True)

    patients = sorted([
        os.path.join(acdc_root, d)
        for d in os.listdir(acdc_root)
        if os.path.isdir(os.path.join(acdc_root, d)) and re.match(r"patient\d+", d)
    ])

    if not patients:
        raise RuntimeError(f"No patient folders found under: {acdc_root}")

    # 保存映射关系，方便你回溯 image_17 是哪个 patient 的哪个 phase
    map_csv = os.path.join(out_folder, "index_mapping.csv")
    with open(map_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["new_index", "patient_id", "phase", "src_img", "src_gt"])

        idx = 1
        for pdir in patients:
            pid = os.path.basename(pdir)

            cfg_path = os.path.join(pdir, "Info.cfg")
            if not os.path.exists(cfg_path):
                print(f"[Skip] {pid}: Info.cfg not found")
                continue

            info = read_info_cfg(cfg_path)
            ed_idx = int(float(info["ED"]))
            es_idx = int(float(info["ES"]))
            if "ED" not in info or "ES" not in info:
                raise RuntimeError(f"{cfg_path} parsed keys: {list(info.keys())}")


            for phase, frame_idx in [("ED", ed_idx), ("ES", es_idx)]:
                img_fid = os.path.join(pdir, frame_name(pid, frame_idx))
                seg_fid = os.path.join(pdir, frame_gt_name(pid, frame_idx))

                if not (os.path.exists(img_fid) and os.path.exists(seg_fid)):
                    print(f"[Skip] {pid} {phase}: frame files not found")
                    continue

                print(f"\n=== Processing {pid} {phase} (frame{frame_idx:02d}) ===")

                img_obj = sitk.ReadImage(img_fid)
                seg_obj = sitk.ReadImage(seg_fid)

                # ---- Step A: histogram bright-end cut ----
                img_arr_zyx = sitk.GetArrayFromImage(img_obj).astype(np.float32)  # (Z,Y,X)
                hir = float(np.percentile(img_arr_zyx, 100.0 - hist_cut_top))
                img_arr_zyx[img_arr_zyx > hir] = hir

                his_img_o = sitk.GetImageFromArray(img_arr_zyx)
                his_img_o = copy_spacing_ori(img_obj, his_img_o)

                # ---- Step B: resample image + label to unified spacing ----
                res_img_o = resample_by_res(his_img_o, list(new_spacing), interpolator=sitk.sitkLinear, logging=logging)
                # label：one-hot resample 再合并（你原来那套）
                res_lb_o  = resample_lb_by_res(seg_obj, list(new_spacing), interpolator=sitk.sitkLinear, ref_img=None, logging=logging)

                # ---- Step C: crop ROI to unified size ----
                res_img_a_zyx = sitk.GetArrayFromImage(res_img_o).astype(np.float32)
                res_lb_a_zyx  = sitk.GetArrayFromImage(res_lb_o).astype(np.int16)

                # fallback: 图像中心 (cy,cx) in (Y,X)
                fallback_xy = (res_img_a_zyx.shape[1] // 2, res_img_a_zyx.shape[2] // 2)
                ctr_yx = compute_xy_center_from_label(res_lb_a_zyx, fallback_xy)

                # 注意：你的 image_crop 期望输入为 (Y,X,Z)，所以这里 transpose(1,2,0)
                img_yxz = res_img_a_zyx.transpose(1, 2, 0)  # (Y,X,Z)
                lb_yxz  = res_lb_a_zyx.transpose(1, 2, 0)

                crop_img_yxz = image_crop(
                    img_yxz,
                    list(crop_hw),
                    referece_ctr_idx=[ctr_yx[0], ctr_yx[1]],
                    padval=float(res_img_a_zyx.min()),
                    only_2d=True
                )
                crop_lb_yxz = image_crop(
                    lb_yxz,
                    list(crop_hw),
                    referece_ctr_idx=[ctr_yx[0], ctr_yx[1]],
                    padval=0,
                    only_2d=True
                )

                # crop 回到 (Z,Y,X) 再写出为 NIfTI
                crop_img_zyx = crop_img_yxz.transpose(2, 0, 1).astype(np.float32)
                crop_lb_zyx  = crop_lb_yxz.transpose(2, 0, 1).astype(np.int16)
                crop_lb_zyx  = np.clip(crop_lb_zyx, 0, 3)  # ACDC 标准 label 0-3

                out_img_obj = sitk.GetImageFromArray(crop_img_zyx)
                out_lb_obj  = sitk.GetImageFromArray(crop_lb_zyx)

                # 用 resampled image 的 spacing/origin/direction（保证一致）
                out_img_obj = copy_spacing_ori(res_img_o, out_img_obj)
                out_lb_obj  = copy_spacing_ori(res_img_o, out_lb_obj)

                out_img_fid = os.path.join(out_folder, f"image_{idx}.nii.gz")
                out_lb_fid  = os.path.join(out_folder, f"label_{idx}.nii.gz")

                sitk.WriteImage(out_img_obj, out_img_fid, True)
                sitk.WriteImage(out_lb_obj,  out_lb_fid,  True)

                writer.writerow([idx, pid, phase, img_fid, seg_fid])
                if logging:
                    print(f"[Saved] {out_img_fid}")
                    print(f"[Saved] {out_lb_fid}")

                idx += 1

    print(f"\nDone. Output folder: {out_folder}")
    print(f"Index mapping saved to: {map_csv}")
