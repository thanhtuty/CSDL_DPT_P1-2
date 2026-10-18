"""
preprocess.py - Chuan hoa bo anh van tay FVC2000 DB1+DB2 ve cung kich thuoc 256x256.

Pipeline:
  1. Phan vung (segmentation) vung van tay bang phuong sai khoi + do nhat quan huong
  2. Tim trong tam vung van tay -> cat cua so 256x256 can giua trong tam
  3. Chuan hoa cuong do sang theo Hong-Wan-Jain (mean/variance normalization)
  4. Luu PNG 8-bit grayscale (khong mat mat)

Ca DB1 va DB2 deu 500 dpi nen cat cung mot kich thuoc vat ly (256 px = 13.0 mm)
giu nguyen chu ky van => tan so van so sanh duoc truc tiep giua hai cam bien.
"""
import os, re, glob, json, math
import numpy as np
import cv2
from config import FVC_DBS, OUT_DATASET, DATABASES, N_FINGERS, SIZE, BLK, M0, V0

SRC, OUT = FVC_DBS, OUT_DATASET


def segment(img, blk=BLK):
    """Tra ve mask nhi phan (uint8 0/1) cua vung van tay, do phan giai pixel."""
    h, w = img.shape
    f = img.astype(np.float32)

    # (a) do lech chuan cuc bo theo khoi
    hb, wb = h // blk, w // blk
    f_crop = f[:hb * blk, :wb * blk]
    blocks = f_crop.reshape(hb, blk, wb, blk).swapaxes(1, 2).reshape(hb, wb, blk * blk)
    std = blocks.std(axis=2)

    # (b) do nhat quan huong (coherence) - vung van co gradient dinh huong manh
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    gxx = cv2.boxFilter(gx * gx, -1, (blk, blk))
    gyy = cv2.boxFilter(gy * gy, -1, (blk, blk))
    gxy = cv2.boxFilter(gx * gy, -1, (blk, blk))
    num = np.sqrt((gxx - gyy) ** 2 + 4 * gxy ** 2)
    den = gxx + gyy + 1e-6
    coh = (num / den)[:hb * blk, :wb * blk].reshape(hb, blk, wb, blk).swapaxes(1, 2).mean(axis=(2, 3))

    # nguong Otsu tren do lech chuan + nguong co dinh tren coherence
    s8 = cv2.normalize(std, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    t, _ = cv2.threshold(s8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = ((s8 >= t) & (coh > 0.30)).astype(np.uint8)

    # lam sach hinh thai hoc + giu thanh phan lien thong lon nhat
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (lab == k).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    full = np.zeros((h, w), np.uint8)
    full[:hb * blk, :wb * blk] = cv2.resize(m, (wb * blk, hb * blk), interpolation=cv2.INTER_NEAREST)
    return full


def center_crop(img, mask, size=SIZE):
    """Cat cua so size x size can giua trong tam vung van tay (pad bang nen neu tran)."""
    h, w = img.shape
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        cy, cx = h // 2, w // 2
    else:
        cy, cx = int(round(ys.mean())), int(round(xs.mean()))

    # gia tri nen = trung binh vung ngoai van tay (de pad khong tao canh gia)
    bg = float(img[mask == 0].mean()) if (mask == 0).any() else float(img.mean())

    half = size // 2
    y0, x0 = cy - half, cx - half
    out = np.full((size, size), bg, np.float32)
    mo = np.zeros((size, size), np.uint8)

    ys0, ys1 = max(0, y0), min(h, y0 + size)
    xs0, xs1 = max(0, x0), min(w, x0 + size)
    dy0, dx0 = ys0 - y0, xs0 - x0
    out[dy0:dy0 + (ys1 - ys0), dx0:dx0 + (xs1 - xs0)] = img[ys0:ys1, xs0:xs1]
    mo[dy0:dy0 + (ys1 - ys0), dx0:dx0 + (xs1 - xs0)] = mask[ys0:ys1, xs0:xs1]
    return out, mo, (cy, cx)


def normalize_intensity(img, mask, m0=M0, v0=V0):
    """Chuan hoa Hong-Wan-Jain: dua trung binh ve m0, phuong sai ve v0 tren vung van tay."""
    f = img.astype(np.float32)
    sel = mask > 0
    if sel.sum() < 100:
        sel = np.ones_like(mask, bool)
    m, v = float(f[sel].mean()), float(f[sel].var())
    v = max(v, 1.0)
    d = np.sqrt(v0 * (f - m) ** 2 / v)
    out = np.where(f > m, m0 + d, m0 - d)
    return np.clip(out, 0, 255).astype(np.uint8)


def run():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for db, tag in DATABASES:
        os.makedirs(f"{OUT}/{tag}", exist_ok=True)
        files = sorted(glob.glob(os.path.join(SRC, db, "*.tif")),
                       key=lambda p: [int(x) for x in re.findall(r"\d+", os.path.basename(p))])
        if N_FINGERS:                       # chi lay N ngon dau tien
            files = [f for f in files
                     if int(re.findall(r"\d+", os.path.basename(f))[0]) <= N_FINGERS]
        for p in files:
            fin, imp = (int(x) for x in re.findall(r"\d+", os.path.basename(p)))
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            h, w = img.shape
            mask = segment(img)
            crop, mcrop, (cy, cx) = center_crop(img, mask, SIZE)
            norm = normalize_intensity(crop, mcrop)

            name = f"{tag}_{fin:03d}_{imp}.png"
            cv2.imwrite(f"{OUT}/{tag}/{name}", norm)
            np.save(f"{OUT}/{tag}/{name[:-4]}_mask.npy", mcrop)

            rows.append(dict(
                file=f"{tag}/{name}", db=tag, finger=fin, impression=imp,
                src_w=w, src_h=h,
                fg_ratio_src=float(mask.mean()),
                fg_ratio_crop=float(mcrop.mean()),
                centroid_y=cy, centroid_x=cx,
                offset_y=cy - h / 2, offset_x=cx - w / 2,
                mean_src=float(img.mean()), std_src=float(img.std()),
                mean_out=float(norm.mean()), std_out=float(norm.std()),
            ))
        print(tag, "done", len(files))

    with open(f"{OUT}/manifest.json", "w") as fh:
        json.dump(rows, fh)
    print("total", len(rows))


if __name__ == "__main__":
    run()
