"""
features.py - Cac bo thuoc tinh nhan dien van tay.

Bo A (Cap 1 - toan cuc) : truong huong, do nhat quan, diem ky di, loai hoa tay
Bo B (Cap 1 - tan so)   : chu ky van, mat do van
Bo C (Cap 2 - cuc bo)   : minutiae (diem ket thuc / diem re nhanh)
Bo D (Ket cau)          : FingerCode (Gabor), LBP, GLCM
Bo E (Chat luong)       : ti le vung van, do nhat quan trung binh, OCL
"""
import numpy as np, cv2, math
from skimage.morphology import skeletonize
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

from config import BLK as W, DPI, MM_PER_PX   # o 16 px, 500 dpi, 0.0508 mm/px


# ---------------------------------------------------------------- Bo A
def orientation_field(img, w=W):
    """Truong huong theo phuong phap gradient (Rao). Tra ve (theta, coherence) kich thuoc (h/w, w/w).

    theta in [0, pi) la huong CUA DUONG VAN (vuong goc voi gradient trung binh).
    """
    f = img.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)

    # tensor cau truc tich luy tren o w x w
    gxx = cv2.boxFilter(gx * gx, -1, (w, w), normalize=True)
    gyy = cv2.boxFilter(gy * gy, -1, (w, w), normalize=True)
    gxy = cv2.boxFilter(gx * gy, -1, (w, w), normalize=True)

    h, wd = img.shape
    hb, wb = h // w, wd // w
    c = w // 2
    ys = np.arange(hb) * w + c
    xs = np.arange(wb) * w + c
    Gxx, Gyy, Gxy = gxx[np.ix_(ys, xs)], gyy[np.ix_(ys, xs)], gxy[np.ix_(ys, xs)]

    # goc gradient chinh: 2*phi = atan2(2Gxy, Gxx-Gyy); huong van = phi + pi/2
    num, den = 2 * Gxy, Gxx - Gyy
    theta = 0.5 * np.arctan2(num, den) + np.pi / 2.0
    theta = np.mod(theta, np.pi)
    coh = np.sqrt(num ** 2 + den ** 2) / (Gxx + Gyy + 1e-6)
    return theta.astype(np.float32), np.clip(coh, 0, 1).astype(np.float32)


def smooth_orientation(theta, coh, k=3):
    """Lam min truong huong bang trung binh vector tren goc gap doi."""
    s = cv2.blur(np.sin(2 * theta) * coh, (k, k))
    c = cv2.blur(np.cos(2 * theta) * coh, (k, k))
    return np.mod(0.5 * np.arctan2(s, c), np.pi).astype(np.float32)


def poincare(theta, mask_b, step=1):
    """Chi so Poincare tren truong huong -> danh sach (y, x, loai).

    loai: +1/2 = core (nhan), -1/2 = delta (tam giac), +1 = whorl kep
    """
    h, w = theta.shape
    out = []
    # duyet vong 8 lan can theo chieu kim dong ho
    dy = [-1, -1, -1, 0, 1, 1, 1, 0]
    dx = [-1, 0, 1, 1, 1, 0, -1, -1]
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if mask_b[i, j] < 0.75:
                continue
            if not mask_b[i - 1:i + 2, j - 1:j + 2].all():
                continue
            tot = 0.0
            for k in range(8):
                a = theta[i + dy[k], j + dx[k]]
                b = theta[i + dy[(k + 1) % 8], j + dx[(k + 1) % 8]]
                d = b - a
                # dua hieu ve (-pi/2, pi/2] vi huong la dai luong mod pi
                while d > np.pi / 2:
                    d -= np.pi
                while d <= -np.pi / 2:
                    d += np.pi
                tot += d
            idx = tot / (2 * np.pi)   # chi so Poincare chuan: +1/2 core, -1/2 delta, +1 whorl
            if 0.45 < idx < 0.55:
                out.append((i, j, 'core'))
            elif -0.55 < idx < -0.45:
                out.append((i, j, 'delta'))
            elif 0.95 < idx < 1.05:
                out.append((i, j, 'whorl'))
    return out


def merge_sp(sps, r=2):
    """Gop cac diem ky di sat nhau cung loai."""
    out = []
    used = [False] * len(sps)
    for i, (y, x, t) in enumerate(sps):
        if used[i]:
            continue
        g = [(y, x)]
        used[i] = True
        for j in range(i + 1, len(sps)):
            if used[j] or sps[j][2] != t:
                continue
            if abs(sps[j][0] - y) <= r and abs(sps[j][1] - x) <= r:
                g.append((sps[j][0], sps[j][1]))
                used[j] = True
        ys = sum(p[0] for p in g) / len(g)
        xs = sum(p[1] for p in g) / len(g)
        out.append((ys, xs, t, len(g)))
    return out


def classify_pattern(sps):
    """Phan loai hoa tay tu so core/delta (quy tac Galton-Henry rut gon)."""
    nc = sum(1 for s in sps if s[2] == 'core')
    nd = sum(1 for s in sps if s[2] == 'delta')
    nw = sum(1 for s in sps if s[2] == 'whorl')
    if nw >= 1 or nc >= 2:
        return 'whorl', nc, nd
    if nc == 1 and nd == 1:
        return 'loop', nc, nd
    if nc == 1 and nd == 0:
        return 'loop', nc, nd           # delta thuong nam ngoai cua so 256x256
    if nc == 0 and nd == 0:
        return 'arch', nc, nd
    return 'other', nc, nd


# ---------------------------------------------------------------- Bo B
def ridge_frequency(img, theta, mask, w=W):
    """Chu ky van theo phuong phap x-signature (Hong, Wan & Jain 1998).

    Tra ve (freq_map, period_px_median) - freq tinh bang chu ky/pixel.
    """
    h, wd = img.shape
    hb, wb = theta.shape
    f = img.astype(np.float32)
    freq = np.zeros((hb, wb), np.float32)
    L, Wl = 32, 16        # chieu dai / rong cua so dinh huong
    for i in range(hb):
        for j in range(wb):
            cy, cx = i * w + w // 2, j * w + w // 2
            if mask[cy, cx] == 0:
                continue
            if cy - L // 2 < 0 or cy + L // 2 >= h or cx - L // 2 < 0 or cx + L // 2 >= wd:
                continue
            ang = theta[i, j]
            # xoay vung anh sao cho duong van nam ngang -> tin hieu bien thien theo cot
            M = cv2.getRotationMatrix2D((cx, cy), np.degrees(ang) + 90, 1.0)
            rot = cv2.warpAffine(f, M, (wd, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
            blk = rot[cy - L // 2:cy + L // 2, cx - Wl // 2:cx + Wl // 2]
            sig = blk.mean(axis=1)          # x-signature doc theo phuong vuong goc van
            # lam tron tin hieu de khu nhieu hat (quan trong voi cam bien dien dung)
            sig = cv2.GaussianBlur(sig.reshape(-1, 1), (1, 5), 1.0).ravel()
            amp = sig.max() - sig.min()
            if amp < 8.0:                   # tuong phan van qua yeu -> bo qua o nay
                continue
            # chi nhan dinh noi bat (cao hon trung binh) de tranh dinh nhieu
            thr = sig.mean()
            pk = [k for k in range(1, len(sig) - 1)
                  if sig[k] > sig[k - 1] and sig[k] >= sig[k + 1] and sig[k] > thr]
            if len(pk) < 2:
                continue
            d = float(np.diff(pk).mean())
            if 4.0 <= d <= 20.0:            # chu ky van hop le tai 500 dpi: 4-20 px
                freq[i, j] = 1.0 / d
    vals = freq[freq > 0]
    med = float(1.0 / np.median(vals)) if len(vals) else float('nan')
    return freq, med


# ---------------------------------------------------------------- Bo C
def gabor_enhance(img, theta, freq, mask, w=W):
    """Tang cuong anh bang bo loc Gabor bam theo huong & tan so cuc bo."""
    h, wd = img.shape
    f = img.astype(np.float32)
    out = np.zeros_like(f)
    vals = freq[freq > 0]
    f0 = float(np.median(vals)) if len(vals) else 1 / 9.0
    sx = sy = 4.0
    # dung ngan hang 16 huong de tiet kiem thoi gian
    nang = 16
    kers = []
    sz = 17
    for a in range(nang):
        ang = a * np.pi / nang
        k = cv2.getGaborKernel((sz, sz), sx, ang + np.pi / 2, 1.0 / f0, 1.0, 0, cv2.CV_32F)
        k -= k.mean()
        kers.append(k)
    filt = [cv2.filter2D(f, cv2.CV_32F, k) for k in kers]
    idx = np.clip((theta / np.pi * nang).astype(int), 0, nang - 1)
    big = cv2.resize(idx.astype(np.uint8), (wd, h), interpolation=cv2.INTER_NEAREST)
    for a in range(nang):
        sel = big == a
        out[sel] = filt[a][sel]
    out[mask == 0] = 0
    return out


def extract_minutiae(enh, mask, theta, w=W, border=12, erode_px=8):
    """Trich minutiae bang Crossing Number tren anh xuong (skeleton)."""
    b = (enh > 0).astype(np.uint8)
    b = cv2.medianBlur(b * 255, 3) // 255
    b[mask == 0] = 0
    sk = skeletonize(b.astype(bool)).astype(np.uint8)

    h, wd = sk.shape
    # mask co ran de bo canh gia o bien vung van; dong thoi loai khung vien anh
    frame = np.zeros_like(mask)
    frame[border:h - border, border:wd - border] = 1
    er = cv2.erode(mask, np.ones((erode_px * 2 + 1, erode_px * 2 + 1), np.uint8)) * frame
    nb = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    mins = []
    ys, xs = np.nonzero(sk)
    for y, x in zip(ys, xs):
        if y < 1 or x < 1 or y >= h - 1 or x >= wd - 1 or er[y, x] == 0:
            continue
        v = [int(sk[y + dy, x + dx]) for dy, dx in nb]
        cn = sum(abs(v[k] - v[(k + 1) % 8]) for k in range(8)) // 2
        if cn == 1:
            mins.append((y, x, 'ending'))
        elif cn == 3:
            mins.append((y, x, 'bifurcation'))

    # loai minutiae gia: cac cap qua gan nhau (<8 px)
    keep, n = [], len(mins)
    bad = set()
    for i in range(n):
        if i in bad:
            continue
        for j in range(i + 1, n):
            if j in bad:
                continue
            if (mins[i][0] - mins[j][0]) ** 2 + (mins[i][1] - mins[j][1]) ** 2 < 64:
                bad.add(i)
                bad.add(j)
    for i in range(n):
        if i not in bad:
            y, x, t = mins[i]
            ang = float(theta[min(y // w, theta.shape[0] - 1), min(x // w, theta.shape[1] - 1)])
            keep.append((int(y), int(x), t, ang))
    return keep, sk


# ---------------------------------------------------------------- Bo D
def fingercode(img, center, mask, nbands=4, nsect=16, nang=8, r0=16, dr=20):
    """FingerCode (Jain et al. 2000): nang luong Gabor theo o quat quanh diem tham chieu."""
    f = img.astype(np.float32)
    f = f - cv2.blur(f, (33, 33))
    kers = []
    for a in range(nang):
        ang = a * np.pi / nang
        k = cv2.getGaborKernel((17, 17), 4.0, ang + np.pi / 2, 9.0, 1.0, 0, cv2.CV_32F)
        k -= k.mean()
        kers.append(k)
    resp = [np.abs(cv2.filter2D(f, cv2.CV_32F, k)) for k in kers]

    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = center
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    aa = np.mod(np.arctan2(yy - cy, xx - cx), 2 * np.pi)

    vec, cov = [], []
    for b in range(nbands):
        lo, hi = r0 + b * dr, r0 + (b + 1) * dr
        for s in range(nsect):
            a0, a1 = s * 2 * np.pi / nsect, (s + 1) * 2 * np.pi / nsect
            sel = (rr >= lo) & (rr < hi) & (aa >= a0) & (aa < a1) & (mask > 0)
            n = int(sel.sum())
            cov.append(n)
            if n < 20:
                vec.extend([0.0] * nang)
                continue
            for a in range(nang):
                v = resp[a][sel]
                vec.append(float(np.abs(v - v.mean()).mean()))
    v = np.asarray(vec, np.float32)
    nz = np.asarray(cov) >= 20
    return v, nz


def lbp_hist(img, mask, P=8, R=2):
    lbp = local_binary_pattern(img, P, R, method='uniform')
    sel = mask > 0
    h, _ = np.histogram(lbp[sel], bins=np.arange(P + 3), density=True)
    return h.astype(np.float32)


def glcm_feats(img, mask):
    g = (img // 8).astype(np.uint8)        # 32 muc xam
    g[mask == 0] = 0
    m = graycomatrix(g, [1, 3], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                     levels=32, symmetric=True, normed=True)
    out = {}
    for p in ('contrast', 'homogeneity', 'energy', 'correlation'):
        out[p] = float(graycoprops(m, p).mean())
    return out


# ---------------------------------------------------------------- Bo E
def quality(mask, coh, mask_b):
    sel = mask_b > 0.5
    return dict(
        fg_ratio=float(mask.mean()),
        coh_mean=float(coh[sel].mean()) if sel.any() else 0.0,
        coh_std=float(coh[sel].std()) if sel.any() else 0.0,
    )
