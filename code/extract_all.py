"""Trich xuat toan bo dac trung tren cac anh van tay da chuan hoa.

Ngoai cac file NumPy/JSON goc, chuong trinh tao feature_values.csv: moi dong la
mot anh; cac vector va danh sach diem duoc luu trong cot JSON de khong mat gia tri.
"""
import os, json, glob, csv, time
import numpy as np, cv2
from multiprocessing import Pool
import features as F
from config import OUT_DATASET as DS, OUT_FEAT as OUT

W = F.W


def _json_cell(value):
    """Ma hoa list/array thanh JSON gon trong mot o CSV."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return json.dumps(_json_safe(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _json_safe(value):
    """Doi NaN/Inf thanh null de JSON hop le va co the doc bang nhieu cong cu."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def process(path):
    name = os.path.basename(path)[:-4]
    db, fin, imp = name.split("_")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Khong doc duoc anh: {path}")
    mask_path = f"{os.path.dirname(path)}/{name}_mask.npy"
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"Khong tim thay mask: {mask_path}; hay chay preprocess.py truoc.")
    mask = np.load(mask_path)

    theta_raw, coh = F.orientation_field(img, W)
    theta = F.smooth_orientation(theta_raw, coh, 3)

    hb, wb = theta.shape
    mask_b = cv2.resize(mask.astype(np.float32), (wb, hb), interpolation=cv2.INTER_AREA)

    # Diem ky di va phan loai hoa tay.
    sps = F.merge_sp(F.poincare(F.smooth_orientation(theta, coh, 5), mask_b))
    sps = [s for s in sps if s[3] >= 1]
    patt, nc, nd = F.classify_pattern(sps)

    # Chu ky/tan so van; freq la ban do tan so o kich thuoc luoi theta.
    freq, period = F.ridge_frequency(img, theta, mask, W)

    # Tang cuong, skeleton va minutiae.
    enh = F.gabor_enhance(img, theta, freq, mask, W)
    mins, sk = F.extract_minutiae(enh, mask, theta, W)
    nend = sum(1 for m in mins if m[2] == 'ending')
    nbif = sum(1 for m in mins if m[2] == 'bifurcation')
    # Moi diem: [y, x, angle_rad, type_code], type 0=ending, 1=bifurcation.
    minutiae = [[int(m[0]), int(m[1]), float(m[3]),
                 0 if m[2] == 'ending' else 1] for m in mins]

    # FingerCode, LBP va GLCM.
    cy, cx = (sps[0][0] * W + W // 2, sps[0][1] * W + W // 2) if sps else (128, 128)
    fc, fc_valid = F.fingercode(img, (cy, cx), mask)
    lbp = F.lbp_hist(img, mask)
    gl = F.glcm_feats(img, mask)

    # Chat luong anh.
    q = F.quality(mask, coh, mask_b)

    # Vector truong huong 16x16 x 2; dung coherence de giam anh huong o huong kem tin cay.
    of_vec = np.concatenate([(np.cos(2 * theta) * coh * mask_b).ravel(),
                             (np.sin(2 * theta) * coh * mask_b).ravel()]).astype(np.float32)

    rec = dict(
        name=name, db=db, finger=int(fin), impression=int(imp),
        pattern=patt, n_core=nc, n_delta=nd, n_sp=len(sps),
        sp=[[float(a), float(b), c, int(d)] for a, b, c, d in sps],
        ridge_period_px=float(period) if np.isfinite(period) else None,
        ridge_period_mm=period * F.MM_PER_PX if np.isfinite(period) else None,
        ridge_freq_valid=int((freq > 0).sum()),
        n_minutiae=len(mins), n_ending=nend, n_bifurcation=nbif,
        minutiae_density=len(mins) / max(mask.sum() / (F.DPI / 25.4) ** 2, 1e-6),  # /mm^2
        minutiae=minutiae,
        lbp=lbp.tolist(), glcm=gl, quality=q,
        fc_valid_sectors=int(fc_valid.sum()),
        skel_ridge_px=int(sk.sum()),
    )
    return rec, of_vec, fc.astype(np.float32), np.asarray(lbp, np.float32), freq.astype(np.float32)


def write_feature_csv(path, recs, orient, fingercode, lbp, ridge_frequency):
    """Ghi mot dong/anh, cac mang va danh sach diem duoc ma hoa JSON trong tung o."""
    fields = [
        "name", "db", "finger", "impression", "pattern", "n_core", "n_delta", "n_sp",
        "singular_points_json", "ridge_period_px", "ridge_period_mm", "ridge_freq_valid",
        "ridge_frequency_map_json", "n_minutiae", "n_ending", "n_bifurcation",
        "minutiae_density_per_mm2", "minutiae_points_json",
        "lbp_histogram_json", "glcm_contrast", "glcm_homogeneity", "glcm_energy",
        "glcm_correlation", "quality_fg_ratio", "quality_coh_mean", "quality_coh_std",
        "fc_valid_sectors", "skel_ridge_px", "orientation_vector_json",
        "fingercode_vector_json",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for i, rec in enumerate(recs):
            row = {k: rec.get(k, "") for k in (
                "name", "db", "finger", "impression", "pattern", "n_core", "n_delta", "n_sp",
                "ridge_period_px", "ridge_period_mm", "ridge_freq_valid", "n_minutiae", "n_ending",
                "n_bifurcation", "fc_valid_sectors", "skel_ridge_px")}
            row.update({
                "singular_points_json": _json_cell(rec["sp"]),
                "ridge_frequency_map_json": _json_cell(ridge_frequency[i]),
                "minutiae_density_per_mm2": rec["minutiae_density"],
                "minutiae_points_json": _json_cell(rec["minutiae"]),
                "lbp_histogram_json": _json_cell(lbp[i]),
                "glcm_contrast": rec["glcm"]["contrast"],
                "glcm_homogeneity": rec["glcm"]["homogeneity"],
                "glcm_energy": rec["glcm"]["energy"],
                "glcm_correlation": rec["glcm"]["correlation"],
                "quality_fg_ratio": rec["quality"]["fg_ratio"],
                "quality_coh_mean": rec["quality"]["coh_mean"],
                "quality_coh_std": rec["quality"]["coh_std"],
                "orientation_vector_json": _json_cell(orient[i]),
                "fingercode_vector_json": _json_cell(fingercode[i]),
            })
            writer.writerow({k: _json_safe(v) for k, v in row.items()})


def main():
    os.makedirs(OUT, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(DS, "DB*", "*.png")))
    if not paths:
        raise FileNotFoundError(f"Khong tim thay anh PNG trong {DS}; hay chay preprocess.py truoc.")
    t0 = time.time()
    with Pool(os.cpu_count()) as p:
        res = p.map(process, paths, chunksize=4)
    print(f"{len(res)} anh trong {time.time()-t0:.0f}s")

    recs = [r[0] for r in res]
    OFV = np.stack([r[1] for r in res])
    FCV = np.stack([r[2] for r in res])
    LBV = np.stack([r[3] for r in res])
    RFV = np.stack([r[4].ravel() for r in res])
    np.save(f"{OUT}/orient.npy", OFV)
    np.save(f"{OUT}/fingercode.npy", FCV)
    np.save(f"{OUT}/lbp.npy", LBV)
    np.save(f"{OUT}/ridge_frequency.npy", RFV)
    with open(f"{OUT}/records.json", "w", encoding="utf-8") as fh:
        json.dump(_json_safe(recs), fh, ensure_ascii=False, allow_nan=False)
    write_feature_csv(f"{OUT}/feature_values.csv", recs, OFV, FCV, LBV, RFV)
    print("orient", OFV.shape, "fingercode", FCV.shape,
          "lbp", LBV.shape, "ridge_frequency", RFV.shape)
    print("Da ghi", f"{OUT}/feature_values.csv")


if __name__ == "__main__":
    main()
