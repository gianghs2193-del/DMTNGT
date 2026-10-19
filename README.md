# DMTNGT: Phân tích yếu tố gây thương vong trong tai nạn giao thông đường bộ Việt Nam

Đồ án môn **Khai thác dữ liệu**. Project tự xây dựng bộ dữ liệu tai nạn giao thông (TNGT) đường bộ toàn quốc từ tin tức công khai. Quy trình gồm thu thập, làm sạch, biến đổi và khai thác để tìm các yếu tố (thời gian, địa điểm, phương tiện, nguyên nhân, loại đường…) liên quan đến số người chết và bị thương.

Toàn bộ quy trình chạy tự động, **không gán nhãn thủ công**.

## Kết quả chính

| Giai đoạn | Số lượng |
|---|---|
| Bài báo thô thu thập từ 4 nguồn | 8.782 |
| Sau làm sạch cơ bản và loại trùng SHA-1 | 8.757 |
| Bài được xác định là tin TNGT (Self-training) | 1.778 |
| Sau loại ngoại lai (bài thống kê tổng hợp) | 1.774 |
| **Vụ tai nạn duy nhất** (gộp bài cùng vụ bằng MinHash-LSH) | **1.450** |
| Bảng đặc trưng cho mô hình | 1.450 × 86 cột |

## Quy trình

```
Báo điện tử (VnExpress, Dân Trí, VietNamNet, Znews)
        │  crawl_tngt_batch.py   - duyệt chuyên mục theo path + phân trang
        ▼
data/tngt_raw.sqlite3            - dữ liệu THÔ, không bao giờ bị sửa
        │  tngt_pipeline.py
        ├─ tngt_clean.py         - làm sạch: thiếu, trùng, nhiễu, ngoài phạm vi
        ├─ tngt_features.py      - trích xuất trường từ văn bản + hàm băm (SHA-1, MinHash-LSH)
        ├─ tngt_ml.py            - lọc tin TNGT bằng học bán giám sát (Self-training)
        └─ tngt_transform.py     - rút trích đặc trưng, chuẩn hóa, mã hóa
        ▼
data/processed/*.csv             - kết quả từng bước
        │  app_tngt.py
        ▼
Ứng dụng Streamlit               - biểu đồ, luật kết hợp, báo cáo chất lượng dữ liệu
```

## 1. Thu thập dữ liệu

| Nguồn | Chuyên mục | Số bài |
|---|---|---|
| VnExpress | `/thoi-su/giao-thong` | 600 |
| Dân Trí | `/thoi-su/giao-thong` | 513 |
| VietNamNet | `/thoi-su/giao-thong` | 5.469 |
| Znews | `/giao-thong` | 2.200 |

- Thư viện: `requests` + `BeautifulSoup` để lấy link, `trafilatura` để tách nội dung chính.
- Lưu bài vào SQLite và HTML gốc (nén gzip) trong `data/raw_html/`.
- **Hàng đợi kiểm tra bằng hàm băm:** băm SHA-1 URL vào tập băm, kiểm tra trùng trong O(1) nên không kéo lại bài đã có.
- **Cơ chế chặn để không treo máy:** giới hạn số trang, số bài, thời gian chạy và dung lượng đĩa. Crawler cũng dừng khi:
  - gặp nhiều lỗi liên tiếp;
  - bị chặn (HTTP 429/403);
  - gặp một chuỗi bài cũ hơn mốc ngày;
  - trang bị lặp lại.

  Có thể dừng tay bằng Ctrl+C hoặc tạo file `data/STOP`. Crawler tôn trọng `robots.txt` và có độ trễ giữa các request.
- **Checkpoint:** dừng giữa chừng thì lần sau chạy tiếp từ chỗ cũ.

## 2. Làm sạch dữ liệu

| Bước | Loại vấn đề | Trước → Sau | Cách xử lý |
|---|---|---|---|
| Trùng URL | Trùng | 8.782 → 8.782 | Giữ 1 bản mỗi URL |
| Thiếu tiêu đề/nội dung | Giá trị thiếu | → 8.779 | Loại bỏ |
| Trang không phải bài báo | Nhiễu | → 8.777 | Loại trang chính sách, giới thiệu… |
| Dòng rác trong nội dung | Nhiễu | (sửa nội dung) | Xóa "Xem thêm", "Tin liên quan", chú thích ảnh, URL |
| Nội dung < 300 ký tự | Nhiễu | → 8.758 | Loại bài video/ảnh, bóc tách lỗi |
| Thiếu sapo / ngày đăng | Giá trị thiếu | → 8.758 | Lấy câu đầu nội dung; khôi phục ngày từ URL |
| Ngày ngoài [2023-01-01, hôm nay] | Nhiễu | → 8.758 | Loại |
| Trùng nội dung | Trùng | → 8.757 | Băm **SHA-1** nội dung đã chuẩn hóa |
| Không phải tin TNGT | Nhiễu | → 1.778 | **Self-training** (xem bên dưới) |
| Ngoại lai | Ngoại lai | → 1.774 | > 40 người chết hoặc > 100 bị thương trong một "vụ" là bài thống kê |
| Nhiều bài cùng một vụ | Trùng | → **1.450 vụ** | **MinHash-LSH** + khối băm + Union-Find |

**Giá trị thiếu ở mức vụ:**
- Biến phân loại (tỉnh, khung giờ, loại đường) được gán nhãn `"Không rõ"`.
- Số người chết và bị thương được điền `0`, kèm cột cờ `thieu_so_chet` / `thieu_so_bi_thuong` để phân biệt với số 0 thật.

### Lọc tin TNGT bằng học bán giám sát (Self-training)

Chuyên mục "giao thông" chứa nhiều tin không phải tai nạn: hạ tầng, xử phạt, chính sách…
1. **Nhãn yếu** sinh tự động bằng luật từ khóa, không gán tay.
2. `SelfTrainingClassifier(LogisticRegression)` học trên TF-IDF (1–2 gram), ngưỡng tin cậy 0,9.
3. Dự đoán out-of-fold (5-fold) để mô hình **sửa được nhãn yếu sai**: lật 73 bài luật nhận nhầm và tìm lại 145 bài luật bỏ sót.

### Gộp bài cùng vụ bằng hàm băm

Một vụ thường được nhiều báo đưa tin. So sánh mọi cặp bài là O(n²), nên dùng:
- **MinHash** (128 hàm băm `h(x) = (a·x + b) mod (2³¹ − 1)` trên shingle 2 từ) và **LSH** (64 band × 2 hàng) để tìm cặp ứng viên.
- **Khối băm** theo (tỉnh, ngày ± 1) và (tỉnh, số chết ≥ 3, cách nhau ≤ 7 ngày).
- Xác nhận bằng **Jaccard** ≥ 0,22, sau đó gom nhóm bằng **Union-Find**.

Kết quả trùng khớp với cách so sánh vét cạn, nhưng chỉ phải so khoảng 7% số cặp.

### Trích xuất số thương vong

Regex chỉ đếm số người khi đi kèm danh từ chỉ người. Cách này loại các trường hợp như "thoát chết", "suýt chết", "lớp 12", "29 Tết", "20 con heo"… để số người chết không bị đếm khống.

## 3. Biến đổi dữ liệu

| Nhóm | Phép biến đổi |
|---|---|
| **Rút trích đặc trưng** | Từ văn bản: tỉnh (chuẩn hóa 63 → 34 tỉnh theo sắp xếp 1/7/2025), ngày, giờ, số chết/bị thương, loại đường, cờ phương tiện / nguyên nhân / kiểu tai nạn / thời tiết, dịp lễ Tết |
| **Đặc trưng suy diễn** | Vùng miền, cuối tuần, ban đêm, mùa mưa, số phương tiện, số nguyên nhân, tổng thương vong, độ dài bài, số báo đưa tin |
| **Chuẩn hóa** | `log(1+x)`, Min-Max [0, 1], Z-score |
| **Mã hóa** | One-hot (vùng, loại đường, khung giờ); ordinal (khung giờ, thứ, mức độ); frequency (tỉnh); sin/cos (giờ, thứ, tháng); binning số người chết |

> ⚠️ Khi dự đoán nhãn `NHAN_co_tu_vong`, **không** dùng `thieu_so_chet`, `so_chet_*`, `chet_*` làm đầu vào vì chúng rò rỉ nhãn.

## 4. Khai thác và trực quan hóa

Ứng dụng Streamlit (`app_tngt.py`) có các tab:

- 🔑 **Hàm băm & trùng lặp:** thống kê SHA-1, MinHash-LSH, số cặp ứng viên so với vét cạn.
- 🧹 **Làm sạch dữ liệu:** bảng từng bước, giá trị thiếu, ví dụ trước/sau, danh sách bài bị loại kèm lý do.
- 🔧 **Biến đổi dữ liệu:** các phép chuẩn hóa và mã hóa, bảng đặc trưng.
- 📊 **Tổng quan**, 🕒 **Thời gian**, 📍 **Địa điểm**, **Yếu tố & thương vong**.
- **Luật kết hợp (Apriori):** tìm tổ hợp yếu tố hay đi cùng vụ có người chết.
- **Dữ liệu & chất lượng.**

Thanh bên cho phép lọc dữ liệu, tải CSV lên và nạp lại dữ liệu mới.

## Cài đặt và chạy

Yêu cầu Python ≥ 3.10.

```bash
git clone https://github.com/gianghs2193-del/DMTNGT.git
cd DMTNGT
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
python -m pip install -r requirements.txt
python -m pip install requests beautifulsoup4 trafilatura   # chỉ cần nếu chạy crawler
```

### Thu thập dữ liệu (không bắt buộc, repo đã có sẵn `data/tngt_raw.sqlite3`)

```bash
python crawl_tngt_batch.py run                                   # kéo tất cả nguồn
python crawl_tngt_batch.py run --sources znews --max-pages 50    # một nguồn, giới hạn trang
python crawl_tngt_batch.py run --stop-before 2023-01-01          # dừng khi gặp bài cũ hơn mốc
python crawl_tngt_batch.py stats                                 # thống kê số bài theo nguồn
python crawl_tngt_batch.py export --out data/articles.csv        # xuất CSV
```

### Chạy toàn bộ xử lý và ghi kết quả

```bash
python tngt_pipeline.py      # đọc data/tngt_raw.sqlite3, ghi data/processed/*.csv (~40 giây)
```

### Mở ứng dụng

```bash
streamlit run app_tngt.py
```

## Cấu trúc thư mục

```
DMTNGT/
├── crawl_tngt_batch.py      # crawler theo path, có checkpoint và cơ chế chặn
├── tngt_pipeline.py         # chạy toàn bộ quy trình xử lý, xuất CSV
├── tngt_clean.py            # làm sạch dữ liệu
├── tngt_features.py         # trích xuất trường, hàm băm SHA-1, MinHash-LSH, Apriori
├── tngt_ml.py               # Self-training lọc tin TNGT
├── tngt_transform.py        # rút trích đặc trưng, chuẩn hóa, mã hóa
├── app_tngt.py              # ứng dụng Streamlit
├── requirements.txt
└── data/
    ├── tngt_raw.sqlite3     # dữ liệu thô
    ├── raw_html/            # HTML gốc (gzip) theo nguồn
    └── processed/
        ├── 01_bai_da_lam_sach.csv        # bài sau làm sạch
        ├── 02_bai_bi_loai_va_ly_do.csv   # bài bị loại + lý do
        ├── 03_vu_tai_nan_sach.csv        # bảng vụ tai nạn
        ├── 04_bang_dac_trung.csv         # bảng đặc trưng số cho mô hình
        ├── 05_bao_cao_lam_sach.csv       # báo cáo từng bước làm sạch
        ├── 06_gia_tri_thieu.csv          # tỷ lệ thiếu và cách xử lý
        └── 07_bao_cao_bien_doi.csv       # báo cáo các phép biến đổi
```

## Hạn chế

- **Thiên lệch truyền thông:** báo chí ưu tiên đưa tin vụ nghiêm trọng. Vì vậy tỷ lệ người chết/vụ trong bộ dữ liệu (~0,9) cao hơn thống kê chính thức (~0,6). Bộ dữ liệu phản ánh **các vụ được đưa tin**, không phải toàn bộ TNGT.
- Số lượng bài giữa các nguồn chênh lệch do khác nhau về phân trang và độ sâu lưu trữ của từng báo.
- Trích xuất bằng regex và từ điển có thể bỏ sót cách diễn đạt lạ. Nhiều vụ không nêu giờ hoặc loại đường.

## Ghi chú

Dữ liệu được thu thập từ các trang báo công khai, chỉ phục vụ mục đích học tập và nghiên cứu. Bản quyền nội dung bài báo thuộc về các tòa soạn tương ứng.
