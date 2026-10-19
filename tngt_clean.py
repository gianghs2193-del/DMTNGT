# -*- coding: utf-8 -*-
"""
tngt_clean.py - LAM SACH DU LIEU cho bo du lieu TNGT (chay truoc va sau buoc trich xuat).

Muc 1 - tren BAI BAO (truoc trich xuat):
  * Du lieu nhieu  : bo trang khong phai bai bao (chinh sach bao mat, trang chuyen muc...),
                     bo noi dung qua ngan (boc tach loi / bai video, anh), xoa dong rac
                     ("Xem them", "Tin lien quan", URL, chu thich anh) trong noi dung.
  * Gia tri thieu  : thieu tieu de/noi dung -> loai; thieu sapo -> lay cau dau cua noi dung;
                     thieu/sai ngay dang -> khoi phuc tu URL (Dan Tri co ma thoi gian trong URL),
                     khong khoi phuc duoc -> loai.
  * Ngoai pham vi  : ngay dang ngoai khoang [tu_ngay, hom nay] -> loai.
  * Du lieu trung  : trung URL; trung noi dung bang ham bam SHA-1 (xem tngt_features).

Muc 2 - tren VU (sau trich xuat + gop trung):
  * Gia tri thieu  : bien phan loai (tinh, khung gio, loai duong) -> nhan "Không rõ";
                     so nguoi chet / bi thuong khong duoc nhac -> 0 kem cot co danh dau
                     thieu_* = 1 (de biet gia tri la dien vao, khong phai do bai bao).
  * Ngoai lai      : so chet > 40 hoac bi thuong > 100 cho MOT vu -> bai thong ke (loai o buoc loc).
Moi buoc ghi vao bao cao (so ban ghi truoc/sau) de dua vao bao cao do an.
"""
import datetime as dt
import re

import numpy as np
import pandas as pd

MIN_CONTENT_CHARS = 300
NOT_ARTICLE_RE = re.compile(
    r"chính sách bảo mật|điều khoản sử dụng|liên hệ quảng cáo|giới thiệu tòa soạn|trắc nghiệm:", re.I)
BOILERPLATE_LINE_RE = re.compile(
    r"(?im)^\s*(?:xem thêm|tin liên quan|đọc thêm|>>|ảnh:|video:|clip:|nguồn:|theo .{0,30}$).*$")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
DANTRI_TS_RE = re.compile(r"-(20\d{2})(\d{2})(\d{2})(\d{2})(\d{2})\d{5}\.htm$")


def _step(report, buoc, loai, truoc, sau, mo_ta):
    report.append({"Bước": buoc, "Loại vấn đề": loai, "Trước": int(truoc), "Sau": int(sau),
                   "Thay đổi": int(sau) - int(truoc), "Cách xử lý": mo_ta})


def _date_from_url(url):
    m = DANTRI_TS_RE.search(str(url))
    if not m:
        return None
    y, mo, d, h, mi = map(int, m.groups())
    try:
        return dt.datetime(y, mo, d, h, mi).isoformat()
    except ValueError:
        return None


def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = BOILERPLATE_LINE_RE.sub("", s)
    s = URL_RE.sub("", s)
    s = re.sub(r"[ \t ]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n", s)
    return s.strip()


def _drop(df, keep, log, ly_do):
    """Giu cac dong keep=True; ghi lai dong bi loai vao log['loai_bo']."""
    gone = df[~keep]
    for u, t, s_ in zip(gone["url"], gone["title"], gone["source"]):
        log["loai_bo"].append({"url": u, "source": s_, "title": t, "ly_do": ly_do})
    return df[keep]


def lam_sach_bai(df, tu_ngay="2023-01-01", report=None, log=None):
    """Lam sach muc bai bao. Tra ve (df_sach, report). log (dict) nhan danh sach bai bi loai
    va vi du truoc/sau de hien thi."""
    report = [] if report is None else report
    log = {} if log is None else log
    log.setdefault("loai_bo", []); log.setdefault("vi_du_noi_dung", []); log.setdefault("vi_du_sapo", [])
    df = df.copy()
    n0 = len(df)

    # 1. Trung URL
    df = df.drop_duplicates("url")
    _step(report, "Trùng URL", "Dữ liệu trùng", n0, len(df), "Giữ 1 bản cho mỗi URL đã chuẩn hóa")

    # 2. Thieu tieu de / noi dung
    n = len(df)
    for c in ("title", "sapo", "content"):
        df[c] = df[c].where(df[c].notna(), "")
    df = _drop(df, (df["title"].str.strip() != "") & (df["content"].str.strip() != ""), log,
               "Thiếu tiêu đề hoặc nội dung")
    _step(report, "Thiếu tiêu đề hoặc nội dung", "Giá trị thiếu", n, len(df),
          "Loại bỏ (không có văn bản để phân tích)")

    # 3. Trang khong phai bai bao
    n = len(df)
    df = _drop(df, ~df["title"].str.contains(NOT_ARTICLE_RE), log, "Trang không phải bài báo")
    _step(report, "Trang không phải bài báo", "Dữ liệu nhiễu", n, len(df),
          "Loại trang chính sách, giới thiệu, trắc nghiệm…")

    # 4. Xoa dong rac trong noi dung
    before = df["content"].str.len().sum()
    raw_content = df["content"]
    df["content"] = df["content"].map(clean_text)
    changed = raw_content.str.len() - df["content"].str.len() > 15
    for t, r_ in zip(df.loc[changed, "title"].head(8), raw_content[changed].head(8)):
        junk = [m.group(0).strip() for m in BOILERPLATE_LINE_RE.finditer(r_)] + URL_RE.findall(r_)
        log["vi_du_noi_dung"].append({"title": t, "da_xoa": " | ".join(j for j in junk if j)[:300]})
    df["title"] = df["title"].str.strip()
    removed_chars = int(before - df["content"].str.len().sum())
    _step(report, "Dòng rác trong nội dung", "Dữ liệu nhiễu", len(df), len(df),
          f"Xóa dòng 'Xem thêm', 'Tin liên quan', chú thích, URL ({removed_chars:,} ký tự)")

    # 5. Noi dung qua ngan
    n = len(df)
    df = _drop(df, df["content"].str.len() >= MIN_CONTENT_CHARS, log,
               f"Nội dung < {MIN_CONTENT_CHARS} ký tự")
    _step(report, f"Nội dung < {MIN_CONTENT_CHARS} ký tự", "Dữ liệu nhiễu", n, len(df),
          "Loại (bài video/ảnh hoặc bóc tách lỗi)")

    # 6. Thieu sapo -> lay cau dau
    miss_sapo = (df["sapo"].str.strip().str.len() < 20)
    old_sapo = df.loc[miss_sapo, "sapo"]
    df.loc[miss_sapo, "sapo"] = df.loc[miss_sapo, "content"].str.split(r"(?<=[.!?])\s", n=1, regex=True).str[0]
    for t, o, n_ in zip(df.loc[miss_sapo, "title"].head(8), old_sapo.head(8), df.loc[miss_sapo, "sapo"].head(8)):
        log["vi_du_sapo"].append({"title": t, "sapo_cu": o, "sapo_moi": n_})
    _step(report, "Thiếu sapo", "Giá trị thiếu", len(df), len(df),
          f"Điền bằng câu đầu của nội dung ({int(miss_sapo.sum())} bài)")

    # 7. Ngay dang thieu / sai -> khoi phuc tu URL
    pub = pd.to_datetime(df["published"], errors="coerce", utc=True)
    bad = pub.isna()
    rec = df.loc[bad, "url"].map(_date_from_url)
    df.loc[bad & rec.reindex(df.index).notna(), "published"] = rec.dropna()
    pub = pd.to_datetime(df["published"], errors="coerce", utc=True)
    n = len(df)
    df = _drop(df, pub.notna(), log, "Thiếu/sai ngày đăng")
    _step(report, "Thiếu/sai ngày đăng", "Giá trị thiếu", n, len(df),
          f"Khôi phục từ URL ({int(rec.notna().sum())} bài), còn lại loại bỏ")

    # 8. Ngoai pham vi thoi gian
    pub = pd.to_datetime(df["published"], errors="coerce", utc=True)
    lo = pd.Timestamp(tu_ngay, tz="UTC")
    hi = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)
    n = len(df)
    df = _drop(df, (pub >= lo) & (pub <= hi), log, "Ngày đăng ngoài phạm vi")
    _step(report, f"Ngày đăng ngoài [{tu_ngay}, hôm nay]", "Dữ liệu nhiễu", n, len(df),
          "Loại (ngày sai hoặc ngoài phạm vi nghiên cứu)")
    return df, report


CAT_COLS = {"tinh_moi": "Không rõ", "tinh_cu": "Không rõ", "loai_duong": "Không rõ"}


def missing_profile(vu, cols):
    return {c: float(vu[c].isna().mean()) if c in vu else np.nan for c in cols}


def xu_ly_thieu_vu(vu):
    """Muc vu: xu ly gia tri thieu sau trich xuat. Tra ve (vu_sach, bang_thieu)."""
    vu = vu.copy()
    vu["loai_duong"] = vu["loai_duong"].replace("khác/không rõ", np.nan)
    cols = ["tinh_moi", "tinh_cu", "ngay_xay_ra", "khung_gio", "loai_duong", "so_chet", "so_bi_thuong"]
    before = missing_profile(vu, cols)

    for c, v in CAT_COLS.items():
        vu[c] = vu[c].fillna(v)
    vu["khung_gio"] = vu["khung_gio"].astype(object).fillna("Không rõ")
    for c in ("so_chet", "so_bi_thuong"):
        vu[f"thieu_{c}"] = vu[c].isna().astype(int)
        vu[c] = vu[c].fillna(0).astype(int)
    vu["co_tu_vong"] = (vu["so_chet"] > 0).astype(int)

    after = {c: float((vu[c].isna() | (vu[c].astype(str) == "Không rõ")).mean()) if c in vu else np.nan
             for c in cols}
    how = {"tinh_moi": "Nhãn \"Không rõ\"", "tinh_cu": "Nhãn \"Không rõ\"",
           "ngay_xay_ra": "Đã thay bằng ngày đăng khi bài không nêu ngày",
           "khung_gio": "Nhãn \"Không rõ\" (không suy đoán giờ)", "loai_duong": "Nhãn \"Không rõ\"",
           "so_chet": "Điền 0 + cột cờ thieu_so_chet", "so_bi_thuong": "Điền 0 + cột cờ thieu_so_bi_thuong"}
    table = pd.DataFrame({"Trường": cols, "% thiếu trước": [before[c] for c in cols],
                          "% 'Không rõ' sau": [after[c] for c in cols],
                          "Cách xử lý": [how[c] for c in cols]})
    return vu, table
