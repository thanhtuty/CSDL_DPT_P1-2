"""
config.py — Cau hinh duong dan va tham so. SUA FILE NAY TRUOC KHI CHAY.

Chi can sua DUY NHAT bien FVC_DBS ben duoi cho khop voi may cua ban.
"""
import os

# ---------------------------------------------------------------- DUONG DAN
# Thu muc "Dbs" cua FVC2000 (chua Db1_a, Db1_b, Db2_a, ...).
# Tren may Kar (Windows) duong dan hien tai la:
FVC_DBS = r"D:\PTIT\CNTT_N4_KY_I\He_csdl_da_phuong_tien\Qly_vantay" \
          r"\74034_3_En_4_MOESM1_ESM\FVC2000\Dbs"

# Thu muc ket qua — se duoc tu dong tao neu chua co.
OUT_ROOT    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ketqua")
OUT_DATASET = os.path.join(OUT_ROOT, "dataset")   # anh da chuan hoa 256x256
OUT_FEAT    = os.path.join(OUT_ROOT, "feat")      # vector dac trung + records.json

# ---------------------------------------------------------------- PHAM VI
# (thu_muc_nguon, nhan_dat_ten). Them/bot dong de doi pham vi bo du lieu.
DATABASES = [("Db1_a", "DB1"), ("Db2_a", "DB2")]

# So ngon tay lay tu moi CSDL (1..N_FINGERS). FVC2000 co toi da 100 ngon o tap _a.
# 65 ngon x 8 lan lan x 2 CSDL = 1.040 anh (dung bang bao cao).
# Dat None de lay toan bo 100 ngon (= 1.600 anh).
N_FINGERS = 65

# ---------------------------------------------------------------- THAM SO
SIZE   = 256      # kich thuoc anh sau chuan hoa (px)
BLK    = 16       # kich thuoc o dung cho segmentation va truong huong
DPI    = 500.0    # do phan giai cua ca DB1 va DB2
M0, V0 = 128.0, 2000.0   # trung binh / phuong sai dich khi chuan hoa cuong do

MM_PER_PX = 25.4 / DPI   # 0.0508 mm moi pixel
