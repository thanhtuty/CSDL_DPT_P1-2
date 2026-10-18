"""extract_all.py - Trich xuat toan bo dac trung tren 1040 anh da chuan hoa."""
import os, json, glob, re, sys, time
import numpy as np, cv2
from multiprocessing import Pool
import features as F
from config import OUT_DATASET as DS, OUT_FEAT as OUT

W = F.W


def process(path):
    name = os.path.basename(path)[:-4]
    db, fin, imp = name.split("_")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = np.load(f"{os.path.dirname(path)}/{name}_mask.npy")

    theta_raw, coh = F.orientation_field(img, W)
    theta = F.smooth_orientation(theta_raw, coh, 3)

    hb, wb = theta.shape
    mask_b = cv2.resize(mask.astype(np.float32), (wb, hb), interpolation=cv2.INTER_AREA)

    # --- Bo A: diem ky di + phan loai
    sps = F.merge_sp(F.poincare(F.smooth_orientation(theta, coh, 5), mask_b))
    sps = [s for s in sps if s[3] >= 1]
    patt, nc, nd = F.classify_pattern(sps)

    # --- Bo B: tan so van
    freq, period = F.ridge_frequency(img, theta, mask, W)

    # --- Bo C: minutiae
    enh = F.gabor_enhance(img, theta, freq, mask, W)
    mins, sk = F.extract_minutiae(enh, mask, theta, W)
    nend = sum(1 for m in mins if m[2] == 'ending')
    nbif = sum(1 for m in mins if m[2] == 'bifurcation')

    # --- Bo D: ket cau
    cy, cx = (sps[0][0] * W + W // 2, sps[0][1] * W + W // 2) if sps else (128, 128)
    fc, fc_valid = F.fingercode(img, (cy, cx), mask)
    lbp = F.lbp_hist(img, mask)
    gl = F.glcm_feats(img, mask)

    # --- Bo E: chat luong
    q = F.quality(mask, coh, mask_b)

    # vector truong huong (cos2t, sin2t) co trong so coherence, 16x16 o -> 512 chieu
    of_vec = np.concatenate([(np.cos(2 * theta) * mask_b).ravel(),
                             (np.sin(2 * theta) * mask_b).ravel()]).astype(np.float32)

    rec = dict(
        name=name, db=db, finger=int(fin), impression=int(imp),
        pattern=patt, n_core=nc, n_delta=nd, n_sp=len(sps),
        sp=[(float(a), float(b), c, int(d)) for a, b, c, d in sps],
        ridge_period_px=period,
        ridge_period_mm=period * F.MM_PER_PX if period == period else float('nan'),
        ridge_freq_valid=int((freq > 0).sum()),
        n_minutiae=len(mins), n_ending=nend, n_bifurcation=nbif,
        minutiae_density=len(mins) / max(mask.sum() / (F.DPI / 25.4) ** 2, 1e-6),  # /mm^2
        lbp=lbp.tolist(), glcm=gl, quality=q,
        fc_valid_sectors=int(fc_valid.sum()),
        skel_ridge_px=int(sk.sum()),
    )
    return rec, of_vec, fc, np.asarray(lbp, np.float32)


def main():
    os.makedirs(OUT, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(DS, "DB*", "*.png")))
    t0 = time.time()
    with Pool(os.cpu_count()) as p:
        res = p.map(process, paths, chunksize=4)
    print(f"{len(res)} anh trong {time.time()-t0:.0f}s")

    recs = [r[0] for r in res]
    OFV = np.stack([r[1] for r in res])
    FCV = np.stack([r[2] for r in res])
    LBV = np.stack([r[3] for r in res])
    np.save(f"{OUT}/orient.npy", OFV)
    np.save(f"{OUT}/fingercode.npy", FCV)
    np.save(f"{OUT}/lbp.npy", LBV)
    with open(f"{OUT}/records.json", "w") as fh:
        json.dump(recs, fh)
    print("orient", OFV.shape, "fingercode", FCV.shape, "lbp", LBV.shape)


if __name__ == "__main__":
    main()
