# Code chuẩn hoá ảnh & trích rút đặc trưng vân tay

Bộ mã nguồn dùng cho Câu 1 (xây dựng bộ dữ liệu) và Câu 2 (bộ thuộc tính) của bài tập lớn
*Hệ CSDL Đa phương tiện — Xây dựng hệ CSDL lưu trữ và tìm kiếm ảnh vân tay*.

---

## 1. Các file

| File | Vai trò |
|---|---|
| `config.py` | **Cấu hình đường dẫn và tham số — sửa file này trước khi chạy.** |
| `preprocess.py` | Chuẩn hoá ảnh: phân vùng → căn giữa trọng tâm → cắt 256×256 → chuẩn hoá cường độ sáng |
| `features.py` | Thư viện định nghĩa toàn bộ đặc trưng (không chạy trực tiếp) |
| `extract_all.py` | Trích rút đặc trưng trên toàn bộ ảnh đã chuẩn hoá, xuất vector + `records.json` |
| `dump_minutiae.py` | Xuất toạ độ + hướng của từng minutia ra `minutiae.npz` |
| `viz.py` | Vẽ trường hướng / điểm kỳ dị / skeleton / minutiae để kiểm chứng bằng mắt |

## 2. Yêu cầu môi trường

```
python >= 3.9
pip install numpy opencv-python scikit-image
```

## 3. Cách chạy

**Bước 0.** Mở `config.py`, sửa biến `FVC_DBS` trỏ tới thư mục `Dbs` của FVC2000 trên máy bạn
(thư mục chứa `Db1_a`, `Db2_a`, …). Mặc định đang trỏ đúng vào thư mục trên máy hiện tại.

**Bước 1 — chuẩn hoá ảnh** (≈ 20 giây cho 1.040 ảnh)

```
python preprocess.py
```

Sinh ra `ketqua/dataset/DB1/`, `ketqua/dataset/DB2/` chứa ảnh PNG 256×256, kèm file mặt nạ
`*_mask.npy` và `ketqua/dataset/manifest.json`.

**Bước 2 — trích rút đặc trưng** (≈ 4 phút cho 1.040 ảnh, chạy đa tiến trình)

```
python extract_all.py
```

Sinh ra trong `ketqua/feat/`:

| File | Nội dung |
|---|---|
| `records.json` | Mỗi ảnh một bản ghi: kiểu hoa tay, số core/delta, chu kỳ vân (px và mm), số minutiae theo loại, LBP, GLCM, chất lượng |
| `orient.npy` | Ma trận (N, 512) — trường hướng, lưới 16×16, dạng `cos2θ ‖ sin2θ` có trọng số |
| `fingercode.npy` | Ma trận (N, 512) — FingerCode: 4 vành × 16 ô quạt × 8 hướng Gabor |
| `lbp.npy` | Ma trận (N, 10) — lược đồ LBP uniform |

**Bước 3 — xuất toạ độ minutiae** (≈ 3 phút, chỉ cần nếu làm so khớp cấp 2)

```
python dump_minutiae.py
```

Sinh ra `ketqua/feat/minutiae.npz`. Mỗi ảnh là một mảng `(n, 4)`: `[y, x, góc_radian, loại]`
với loại `0` = điểm kết thúc vân, `1` = điểm rẽ nhánh.

**Kiểm chứng bằng mắt** (tuỳ chọn)

```python
import cv2, viz
a = viz.analyse("ketqua/dataset/DB1/DB1_001_1.png")
cv2.imwrite("kiemtra.png", viz.panel(a))
print(a["sps"], a["per"], len(a["mins"]))
```

Sinh ảnh 4 ô: trường hướng + điểm kỳ dị | ảnh tăng cường Gabor | skeleton | minutiae.

## 4. Đổi phạm vi bộ dữ liệu

Trong `config.py`:

- `N_FINGERS = 65` → lấy ngón 1–65 mỗi CSDL = **1.040 ảnh** (đúng con số trong báo cáo).
  Đặt `N_FINGERS = None` để lấy toàn bộ 100 ngón = 1.600 ảnh.
- `DATABASES = [("Db1_a", "DB1"), ("Db2_a", "DB2")]` → thêm dòng `("Db3_a", "DB3")` nếu
  muốn dùng thêm CSDL khác.

> **Lưu ý khi thêm DB3/DB4.** Code cắt một cửa sổ cố định 256 px = 13,0 mm vì DB1 và DB2
> đều ở 500 dpi. DB4 chỉ "khoảng 500 dpi" nên tỉ lệ px/mm sẽ lệch nhẹ, làm chu kỳ vân
> không còn so sánh trực tiếp được với DB1/DB2.

## 5. Ghi chú kỹ thuật

Ba chỗ dễ sai nhất khi cài lại từ đầu, đã xử lý sẵn trong code:

1. **Trường hướng phải cộng dồn tensor cấu trúc, không lấy trung bình góc.** Hướng vân là đại
   lượng modulo π: trung bình của 179° và 1° cho 90°, tức vuông góc với cả hai. Làm việc trên
   góc nhân đôi (`cos2θ`, `sin2θ`) khử được nghịch lý này.
2. **Chỉ số Poincaré chia cho 2π, không chia cho π.** Core = +1/2, delta = −1/2, whorl = +1.
   Chia nhầm hệ số sẽ khiến mọi core bị gán nhãn whorl.
3. **x-signature phải làm trơn trước khi tìm đỉnh.** Nhiễu hạt của cảm biến điện dung tạo đỉnh
   giả xen giữa các đỉnh thật, làm chu kỳ đo được giảm một nửa (4,2 px thay vì 8,2 px).

## 6. Số liệu tham chiếu

Chạy với `N_FINGERS = 65` trên FVC2000 DB1 + DB2 cho kết quả sau (để đối chiếu khi chạy lại):

| Chỉ tiêu | DB1 | DB2 |
|---|---|---|
| Số ảnh | 520 | 520 |
| Chu kỳ vân | 10,07 ± 0,67 px = 0,512 mm | 8,15 ± 0,46 px = 0,414 mm |
| Số minutiae/ảnh | 35,2 ± 11,9 | 38,6 ± 11,5 |
| Độ nhất quán hướng | 0,733 ± 0,061 | 0,642 ± 0,084 |
| Tỉ lệ vùng vân tay | 92,6 % | 98,9 % |

Phân bố hoa tay trên cả 1.040 ảnh: loop 655, whorl 177, arch 177, khác 31.
Tổng số minutiae trích được: 38.371.
