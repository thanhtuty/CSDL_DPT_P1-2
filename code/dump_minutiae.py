"""dump_minutiae.py - Luu toa do + huong cua minutiae cho tung anh (phuc vu so khop cap 2)."""
import os, glob, json, time
import numpy as np, cv2
from multiprocessing import Pool
import features as F
from config import OUT_DATASET, OUT_FEAT

W = F.W


def one(path):
    name = os.path.basename(path)[:-4]
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = np.load(f"{os.path.dirname(path)}/{name}_mask.npy")
    th_raw, coh = F.orientation_field(img, W)
    th = F.smooth_orientation(th_raw, coh, 3)
    freq, per = F.ridge_frequency(img, th, mask, W)
    enh = F.gabor_enhance(img, th, freq, mask, W)
    mins, sk = F.extract_minutiae(enh, mask, th, W)
    arr = np.array([[m[0], m[1], m[3], 0.0 if m[2] == 'ending' else 1.0] for m in mins],
                   dtype=np.float32) if mins else np.zeros((0, 4), np.float32)
    return name, arr


def main():
    os.makedirs(OUT_FEAT, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(OUT_DATASET, "DB*", "*.png")))
    t0 = time.time()
    with Pool(os.cpu_count()) as p:
        res = p.map(one, paths, chunksize=4)
    print(f"{len(res)} anh / {time.time()-t0:.0f}s")
    np.savez_compressed(os.path.join(OUT_FEAT, "minutiae.npz"), **{n: a for n, a in res})
    print("tong minutiae", sum(len(a) for _, a in res))


if __name__ == "__main__":
    main()
