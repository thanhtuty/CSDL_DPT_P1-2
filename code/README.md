# Code chuẩn hóa ảnh và trích rút đặc trưng vân tay

Bộ mã nguồn dùng cho Câu 1 (xây dựng bộ dữ liệu) và Câu 2 (bộ thuộc tính) của bài tập lớn *Hệ CSDL Đa phương tiện — Xây dựng hệ CSDL lưu trữ và tìm kiếm ảnh vân tay*.

## 1. Các file

| File | Vai trò |
|---|---|
| `config.py` | Cấu hình đường dẫn dữ liệu và tham số; cần sửa `FVC_DBS` trước khi chạy. |
| `preprocess.py` | Phân vùng, căn giữa, cắt ảnh 256×256, chuẩn hóa cường độ sáng. |
| `features.py` | Thư viện hàm trích xuất; không chạy trực tiếp. |
| `extract_all.py` | Trích xuất đặc trưng trên toàn bộ ảnh chuẩn hóa; lưu vector, bản ghi JSON và CSV đầy đủ. |
| `dump_minutiae.py` | Xuất tọa độ và hướng minutiae ra `minutiae.npz` khi cần đối sánh riêng. |
| `viz.py` | Trực quan hóa trường hướng, điểm kỳ dị, Gabor, skeleton và minutiae. |

## 2. Môi trường

Yêu cầu Python >= 3.9 và các thư viện:

```bash
pip install numpy opencv-python scikit-image
```

## 3. Cách chạy

Mở `config.py` và sửa `FVC_DBS` thành đường dẫn tới thư mục `Dbs` của FVC2000 trên máy đang chạy. Thư mục này cần chứa `Db1_a`, `Db2_a`, …

Chạy trong thư mục `code`:

```bash
python preprocess.py
python extract_all.py
```

Nếu cần file minutiae dạng một mảng điểm cho từng ảnh, chạy thêm:

```bash
python dump_minutiae.py
```

`extract_all.py` đọc ảnh PNG và mask do `preprocess.py` tạo. Nếu thiếu ảnh PNG hoặc mask, chương trình báo lỗi và yêu cầu chạy bước tiền xử lý trước.

## 4. Đầu ra

Các kết quả được ghi trong `ketqua/`:

### `ketqua/dataset/`

- `DB1/`, `DB2/`: ảnh xám PNG 256×256.
- `*_mask.npy`: mask nhị phân tương ứng từng ảnh.
- `manifest.json`: thông tin nguồn, kích thước, vùng vân và trọng tâm.

### `ketqua/feat/`

| Tệp | Nội dung |
|---|---|
| `feature_values.csv` | Một dòng cho mỗi ảnh, có các thuộc tính scalar và toàn bộ giá trị vector/điểm. Các vector được lưu thành chuỗi JSON trong từng ô để giữ đủ số chiều. |
| `records.json` | Bản ghi cấu trúc cho mỗi ảnh: hoa tay, tọa độ core/delta, chu kỳ, minutiae, LBP, GLCM và chất lượng. |
| `orient.npy` | Ma trận `(N, 512)`, trường hướng gồm `cos(2θ)` và `sin(2θ)` có trọng số coherence và mask. |
| `fingercode.npy` | Ma trận `(N, 512)`: 4 vành × 16 ô quạt × 8 hướng Gabor. |
| `lbp.npy` | Ma trận `(N, 10)`: histogram LBP uniform. |
| `ridge_frequency.npy` | Ma trận `(N, 256)`: bản đồ tần số đường vân 16×16 được làm phẳng theo từng ảnh. |
| `minutiae.npz` | Tệp tùy chọn từ `dump_minutiae.py`; mỗi ảnh chứa mảng `(n, 4)` dạng `[y, x, góc_radian, loại]`, trong đó loại 0 là ending, 1 là bifurcation. |

Trong `feature_values.csv`, các cột JSON là:

- `singular_points_json`: danh sách điểm kỳ dị `[y, x, loại, số_điểm_gộp]`.
- `minutiae_points_json`: danh sách minutiae `[y, x, góc_radian, mã_loại]`.
- `orientation_vector_json`: vector trường hướng 512 giá trị.
- `fingercode_vector_json`: vector FingerCode 512 giá trị.
- `lbp_histogram_json`: histogram LBP 10 giá trị.
- `ridge_frequency_map_json`: bản đồ tần số 256 giá trị.

Các giá trị có thể được đọc như JSON, ví dụ trong Python:

```python
import json
import pandas as pd

df = pd.read_csv("ketqua/feat/feature_values.csv")
orientation = json.loads(df.loc[0, "orientation_vector_json"])
minutiae = json.loads(df.loc[0, "minutiae_points_json"])
print(df.loc[0, "name"], len(orientation), len(minutiae))
```

Không dùng `feature_values.csv` để lưu riêng các tọa độ minutiae thành hàng độc lập; nếu cần bảng điểm cho truy vấn, có thể đọc `minutiae_points_json` hoặc dùng `minutiae.npz`.

## 5. Đổi phạm vi dữ liệu

Trong `config.py`:

- `N_FINGERS = 65`: lấy 65 ngón đầu tiên của mỗi CSDL, tổng cộng 1.040 ảnh.
- `N_FINGERS = None`: lấy toàn bộ 100 ngón mỗi CSDL, tổng cộng 1.600 ảnh.
- `DATABASES = [("Db1_a", "DB1"), ("Db2_a", "DB2")]`: có thể thêm CSDL khác nếu phù hợp.

DB1 và DB2 có độ phân giải 500 dpi nên 256 px tương ứng khoảng 13,0 mm. DB4 chỉ có độ phân giải xấp xỉ 500 dpi, vì vậy chu kỳ vân theo mm có thể lệch nhẹ nếu thêm nguồn đó.

## 6. Ghi chú kỹ thuật

1. Trường hướng là đại lượng modulo π; làm trơn bằng `cos(2θ)` và `sin(2θ)` để tránh lấy trung bình góc sai tại biên 0°/180°.
2. Chỉ số Poincaré được chia cho `2π`: core xấp xỉ +1/2, delta xấp xỉ −1/2, whorl xấp xỉ +1.
3. Tín hiệu x-signature được làm trơn trước khi tìm đỉnh để giảm đỉnh giả do nhiễu.
4. Vector trường hướng hiện nhân `cos(2θ)` và `sin(2θ)` với coherence và mask. Các file `orient.npy` và `feature_values.csv` dùng cùng cách tính.

## 7. Số liệu tham chiếu

Khi dùng `N_FINGERS = 65` trên FVC2000 DB1 + DB2, bộ dữ liệu gồm 520 ảnh mỗi nguồn, tổng 1.040 ảnh. README ban đầu ghi các số liệu tham chiếu về chu kỳ vân, minutiae, coherence và tỉ lệ vùng vân; kết quả thực tế có thể thay đổi theo phiên bản thư viện và môi trường chạy.
