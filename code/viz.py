"""viz.py - Ve truong huong / diem ky di / minutiae de kiem chung truc quan."""
import numpy as np, cv2, os
import features as F
W = F.W


def analyse(path):
    name = os.path.basename(path)[:-4]
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = np.load(f"{os.path.dirname(path)}/{name}_mask.npy")
    th_raw, coh = F.orientation_field(img, W)
    th = F.smooth_orientation(th_raw, coh, 3)
    hb, wb = th.shape
    mb = cv2.resize(mask.astype(np.float32), (wb, hb), interpolation=cv2.INTER_AREA)
    sps = F.merge_sp(F.poincare(F.smooth_orientation(th, coh, 5), mb))
    freq, per = F.ridge_frequency(img, th, mask, W)
    enh = F.gabor_enhance(img, th, freq, mask, W)
    mins, sk = F.extract_minutiae(enh, mask, th, W)
    return dict(img=img, mask=mask, th=th, coh=coh, mb=mb, sps=sps,
                freq=freq, per=per, enh=enh, mins=mins, sk=sk, name=name)


def panel(a):
    """Tra ve anh BGR gom 4 o: goc+huong | nhi phan hoa | skeleton | minutiae."""
    img, th, mb, sps, mins, sk, enh = (a['img'], a['th'], a['mb'], a['sps'],
                                       a['mins'], a['sk'], a['enh'])
    h, w = img.shape

    # (1) truong huong
    p1 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    hb, wb = th.shape
    for i in range(hb):
        for j in range(wb):
            if mb[i, j] < 0.6:
                continue
            cy, cx = i * W + W // 2, j * W + W // 2
            L = 7
            # theta do trong he toa do anh (y huong xuong) -> ve truc tiep, khong lat y
            dy, dx = L * np.sin(th[i, j]), L * np.cos(th[i, j])
            cv2.line(p1, (int(cx - dx), int(cy - dy)), (int(cx + dx), int(cy + dy)),
                     (0, 200, 255), 1, cv2.LINE_AA)
    for (sy, sx, t, n) in sps:
        cy, cx = int(sy * W + W // 2), int(sx * W + W // 2)
        if t == 'core':
            cv2.circle(p1, (cx, cy), 9, (0, 0, 255), 2)
        elif t == 'delta':
            cv2.drawMarker(p1, (cx, cy), (0, 255, 0), cv2.MARKER_TRIANGLE_UP, 18, 2)
        else:
            cv2.circle(p1, (cx, cy), 11, (255, 0, 255), 2)
            cv2.circle(p1, (cx, cy), 5, (255, 0, 255), 2)

    # (2) anh tang cuong Gabor
    e = enh.copy()
    e = cv2.normalize(e, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    p2 = cv2.cvtColor(e, cv2.COLOR_GRAY2BGR)

    # (3) skeleton
    p3 = cv2.cvtColor((255 - sk * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # (4) minutiae
    p4 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for (y, x, t, ang) in mins:
        c = (0, 0, 255) if t == 'ending' else (255, 128, 0)
        cv2.circle(p4, (x, y), 4, c, 1, cv2.LINE_AA)
        L = 10
        cv2.line(p4, (x, y), (int(x + L * np.cos(ang)), int(y + L * np.sin(ang))), c, 1, cv2.LINE_AA)

    row = np.hstack([p1, p2, p3, p4])
    bar = np.full((22, row.shape[1], 3), 255, np.uint8)
    for k, lab in enumerate(['Truong huong + diem ky di', 'Tang cuong Gabor',
                             'Xuong (skeleton)', 'Minutiae']):
        cv2.putText(bar, lab, (k * w + 4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return np.vstack([row, bar])
