# -*- coding: utf-8 -*-
"""
tngt_transform.py - BIEN DOI DU LIEU: rut trich dac trung, chuan hoa, ma hoa.
Dau vao: bang VU da lam sach (tngt_clean.xu_ly_thieu_vu). Dau ra: bang dac trung san sang cho
mo hinh (moi cot la so) + bao cao tung phep bien doi.

1. Rut trich dac trung (feature extraction)
   - Tu van ban (da lam trong tngt_features): tinh, ngay/gio xay ra, so chet/bi thuong,
     loai duong, phuong tien, nguyen nhan, kieu tai nan, thoi tiet.
   - Dac trung suy dien: vung mien, cuoi tuan, ban dem, mua mua, so phuong tien, so nguyen nhan,
     tong thuong vong, do dai bai, so bao dua tin.
2. Chuan hoa (normalization)
   - Gia tri: ten tinh (bien the -> ten chuan, 63 -> 34 tinh), ngay -> ISO, gio -> khung gio.
   - So: log(1+x) cho bien lech phai, Min-Max ve [0, 1], Z-score (trung binh 0, do lech 1).
3. Ma hoa (encoding)
   - One-hot: vung mien, loai duong, khung gio.
   - Thu tu (ordinal): khung gio, thu trong tuan, muc do nghiem trong.
   - Tan suat (frequency) cho tinh (34 gia tri -> 1 cot, tranh ma tran thua).
   - Chu ky sin/cos cho gio, thu, thang (23h va 0h gan nhau).
   - Roi rac hoa (binning) so nguoi chet -> nhom 0 / 1 / 2 / 3-5 / >5 cho luat ket hop.
"""
import numpy as np
import pandas as pd

import tngt_features as F

REGION = {
    "Bắc": ["Hà Nội", "Hải Phòng", "Quảng Ninh", "Bắc Ninh", "Hưng Yên", "Ninh Bình", "Phú Thọ",
            "Thái Nguyên", "Tuyên Quang", "Lào Cai", "Lai Châu", "Điện Biên", "Sơn La", "Lạng Sơn",
            "Cao Bằng"],
    "Trung": ["Thanh Hóa", "Nghệ An", "Hà Tĩnh", "Quảng Trị", "Huế", "Đà Nẵng", "Quảng Ngãi",
              "Gia Lai", "Khánh Hòa", "Đắk Lắk", "Lâm Đồng"],
    "Nam": ["TP.HCM", "Đồng Nai", "Tây Ninh", "Cần Thơ", "Vĩnh Long", "Đồng Tháp", "Cà Mau", "An Giang"],
}
PROV2REGION = {p: r for r, ps in REGION.items() for p in ps}
SEVERITY_ORDER = ["Không rõ/không thương vong", "Chỉ bị thương", "Có người chết"]
DEATH_BINS = [-0.5, 0.5, 1.5, 2.5, 5.5, np.inf]
DEATH_LABELS = ["0", "1", "2", "3-5", ">5"]


def _report(rows, buoc, nhom, cot, mo_ta):
    rows.append({"Nhóm": nhom, "Phép biến đổi": buoc, "Cột tạo ra": cot, "Mô tả": mo_ta})


def bien_doi(vu):
    """Tra ve (X: bang dac trung so, vu_ext: vu + cot suy dien, report: DataFrame, scale_demo)."""
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    rows = []
    d = vu.copy()
    flag_cols = [c for c in d.columns if c.startswith(("pt_", "nn_", "kieu_", "tt_"))]

    # ---------------- 1. Rut trich dac trung ----------------
    _report(rows, "Trích xuất từ văn bản (regex + từ điển)", "1. Rút trích đặc trưng",
            f"tinh_moi, ngay_xay_ra, gio, so_chet, so_bi_thuong, loai_duong + {len(flag_cols)} cờ 0/1",
            "Tỉnh, thời gian, thương vong, loại đường, phương tiện, nguyên nhân, kiểu tai nạn, thời tiết")
    d["vung"] = d["tinh_moi"].map(PROV2REGION).fillna("Không rõ")
    d["cuoi_tuan"] = d["thu"].astype(str).isin(["Thứ 7", "Chủ nhật"]).astype(int)
    d["ban_dem"] = d["khung_gio"].astype(str).isin([F.TIME_BUCKETS[3], F.TIME_BUCKETS[4]]).astype(int)
    thang = d["ngay_xay_ra"].dt.month
    d["mua_mua"] = thang.between(5, 10).astype(int)
    d["so_phuong_tien"] = d[[c for c in flag_cols if c.startswith("pt_")]].sum(axis=1)
    d["so_nguyen_nhan"] = d[[c for c in flag_cols if c.startswith("nn_")]].sum(axis=1)
    d["tong_thuong_vong"] = d["so_chet"] + d["so_bi_thuong"]
    d["do_dai_bai"] = d["content"].fillna("").str.split().str.len()
    _report(rows, "Đặc trưng suy diễn", "1. Rút trích đặc trưng",
            "vung, cuoi_tuan, ban_dem, mua_mua, so_phuong_tien, so_nguyen_nhan, tong_thuong_vong, "
            "do_dai_bai, so_bai",
            "Vùng miền theo tỉnh; cuối tuần; tối/đêm (18–4h); mùa mưa (tháng 5–10); đếm phương tiện, "
            "nguyên nhân; tổng thương vong; số từ của bài; số bài báo đưa tin (mức độ được chú ý)")

    # ---------------- 2. Chuan hoa ----------------
    _report(rows, "Chuẩn hóa văn bản và giá trị", "2. Chuẩn hóa",
            "title, content, tinh_cu → tinh_moi, ngay_xay_ra, khung_gio",
            "Unicode NFC, chữ thường khi so khớp; biến thể tên tỉnh → tên chuẩn, 63 → 34 tỉnh; "
            "\"tối 15/10\", \"đêm qua\" → ngày ISO; giờ → 5 khung giờ")
    num_cols = ["so_chet", "so_bi_thuong", "tong_thuong_vong", "so_bai", "do_dai_bai",
                "so_phuong_tien", "so_nguyen_nhan"]
    num = d[num_cols].astype(float)
    logged = np.log1p(num)
    _report(rows, "Biến đổi log(1 + x)", "2. Chuẩn hóa", ", ".join(f"{c}_log" for c in num_cols),
            "Giảm độ lệch phải (đa số vụ 0–1 người chết, vài vụ 10 người)")
    mm = pd.DataFrame(MinMaxScaler().fit_transform(logged), columns=[f"{c}_minmax" for c in num_cols],
                      index=d.index)
    _report(rows, "Min-Max scaling", "2. Chuẩn hóa", ", ".join(mm.columns),
            "x' = (x − min) / (max − min), đưa về [0, 1] (sau log)")
    zs = pd.DataFrame(StandardScaler().fit_transform(logged), columns=[f"{c}_z" for c in num_cols],
                      index=d.index)
    _report(rows, "Z-score (StandardScaler)", "2. Chuẩn hóa", ", ".join(zs.columns),
            "z = (x − μ) / σ, trung bình 0, độ lệch chuẩn 1 (cho K-Means, hồi quy…)")

    # ---------------- 3. Ma hoa ----------------
    oh = pd.get_dummies(d[["vung", "loai_duong"]].astype(str), prefix=["vung", "duong"], dtype=int)
    oh_gio = pd.get_dummies(d["khung_gio"].astype(str), prefix="gio", dtype=int)
    oh = pd.concat([oh, oh_gio], axis=1)
    _report(rows, "One-hot encoding", "3. Mã hóa", f"{oh.shape[1]} cột (vung_*, duong_*, gio_*)",
            "Biến định danh không có thứ tự → mỗi giá trị một cột 0/1")
    gio_map = {b: i for i, b in enumerate(F.TIME_BUCKETS)}
    ordn = pd.DataFrame({
        "khung_gio_ord": d["khung_gio"].astype(str).map(gio_map).fillna(-1).astype(int),
        "thu_ord": d["thu"].astype(str).map({w: i for i, w in enumerate(F.WEEKDAYS)}).fillna(-1).astype(int),
        "muc_do_ord": d["muc_do"].map({m: i for i, m in enumerate(SEVERITY_ORDER)}).fillna(0).astype(int),
    }, index=d.index)
    _report(rows, "Ordinal encoding", "3. Mã hóa", "khung_gio_ord, thu_ord, muc_do_ord",
            "Biến có thứ tự → số nguyên (khung giờ 0–4, thứ 0–6, mức độ 0 = không rõ, 1 = bị thương, "
            "2 = có người chết); −1 = không rõ")
    freq = d["tinh_moi"].map(d["tinh_moi"].value_counts(normalize=True)).rename("tinh_tan_suat")
    _report(rows, "Frequency encoding", "3. Mã hóa", "tinh_tan_suat",
            "Tỉnh (34 giá trị) → tỷ lệ số vụ của tỉnh đó, tránh tạo 34 cột one-hot thưa")
    gio = d["gio"]
    cyc = pd.DataFrame({
        "gio_sin": np.where(gio.notna(), np.sin(2 * np.pi * gio.fillna(0) / 24), 0),
        "gio_cos": np.where(gio.notna(), np.cos(2 * np.pi * gio.fillna(0) / 24), 0),
        "thu_sin": np.sin(2 * np.pi * ordn["thu_ord"].clip(lower=0) / 7),
        "thu_cos": np.cos(2 * np.pi * ordn["thu_ord"].clip(lower=0) / 7),
        "thang_sin": np.sin(2 * np.pi * (thang.fillna(1) - 1) / 12),
        "thang_cos": np.cos(2 * np.pi * (thang.fillna(1) - 1) / 12),
    }, index=d.index)
    _report(rows, "Mã hóa chu kỳ sin/cos", "3. Mã hóa", ", ".join(cyc.columns),
            "Giờ, thứ, tháng là biến vòng: sin/cos giữ được việc 23h gần 0h, Chủ nhật gần Thứ 2")
    d["nhom_so_chet"] = pd.cut(d["so_chet"], DEATH_BINS, labels=DEATH_LABELS).astype(str)
    oh_bin = pd.get_dummies(d["nhom_so_chet"], prefix="chet", dtype=int)
    _report(rows, "Rời rạc hóa (binning)", "3. Mã hóa", "nhom_so_chet → chet_0, chet_1, chet_2, chet_3-5, chet_>5",
            "Số người chết → nhóm, dùng cho luật kết hợp (Apriori cần biến rời rạc)")

    binary = d[flag_cols + ["cuoi_tuan", "ban_dem", "mua_mua", "dip_le_tet",
                            "thieu_so_chet", "thieu_so_bi_thuong"]].astype(int)
    _report(rows, "Biến nhị phân giữ nguyên", "3. Mã hóa", f"{binary.shape[1]} cột 0/1",
            "Cờ phương tiện, nguyên nhân, kiểu tai nạn, thời tiết, cuối tuần, ban đêm, lễ Tết, cờ thiếu. "
            "Lưu ý: không dùng thieu_so_chet, so_chet_*, chet_* làm đầu vào khi dự đoán nhãn có tử vong "
            "(rò rỉ nhãn)")

    X = pd.concat([d[["vu_id"]], d[["co_tu_vong"]].rename(columns={"co_tu_vong": "NHAN_co_tu_vong"}),
                   binary, oh, ordn, freq, cyc, logged.add_suffix("_log"), mm, zs, oh_bin], axis=1)
    demo = pd.concat([num["so_chet"].rename("Gốc"), logged["so_chet"].rename("log(1+x)"),
                      mm["so_chet_minmax"].rename("Min-Max"), zs["so_chet_z"].rename("Z-score")], axis=1)
    rep = pd.DataFrame(rows)
    return X, d, rep, demo
