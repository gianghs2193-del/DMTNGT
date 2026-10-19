# -*- coding: utf-8 -*-
"""
tngt_pipeline.py - Chay TOAN BO quy trinh xu ly va (tuy chon) GHI KET QUA RA FILE.

  du lieu tho (SQLite)
    -> lam sach muc bai (tngt_clean)          : trung URL, thieu, nhieu, ngoai pham vi
    -> trung noi dung bang SHA-1 (tngt_features)
    -> trich xuat truong tu van ban (tngt_features)
    -> loc bai TNGT tu dong (tngt_ml, Self-training) + loai ngoai lai
    -> gop bai cung vu bang MinHash-LSH (tngt_features)
    -> xu ly gia tri thieu muc vu (tngt_clean)
    -> bien doi: rut trich dac trung, chuan hoa, ma hoa (tngt_transform)

Chay doc lap (khong can app):
    python tngt_pipeline.py                 # doc data/tngt_raw.sqlite3, ghi data/processed/*.csv
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

import tngt_clean as C
import tngt_features as F
import tngt_ml as M
import tngt_transform as TR

MODULES = [C.__file__, F.__file__, M.__file__, TR.__file__, __file__]


def code_version():
    """Doi khi bat ky file xu ly nao thay doi -> dung lam khoa cache trong app."""
    return tuple(round(os.path.getmtime(f), 2) for f in MODULES)


def doc_sqlite(path):
    uri = "file:" + os.path.abspath(path).replace("\\", "/") + "?mode=ro"
    con = sqlite3.connect(uri, uri=True, timeout=30)
    try:
        return pd.read_sql_query("SELECT source, url, published, title, sapo, content FROM articles", con)
    finally:
        con.close()


def chay(df):
    """Tra ve dict ket qua cua tung buoc."""
    for c in ("source", "url", "published", "title", "sapo", "content"):
        if c not in df:
            df[c] = None
    n_raw = len(df.drop_duplicates("url"))
    log = {}
    df, report = C.lam_sach_bai(df, log=log)

    # Trung noi dung - ham bam SHA-1
    hashes = [F.content_hash(t, c) for t, c in zip(df["title"], df["content"])]
    grp = df.assign(ma_bam=hashes)
    grp = grp[grp.duplicated("ma_bam", keep=False)][["ma_bam", "source", "title", "url"]].sort_values("ma_bam")
    n = len(df)
    df_h, n_hash = F.loai_trung_hash(df)
    for u, t, s in zip(*[df.loc[~df.index.isin(df_h.index), k] for k in ("url", "title", "source")]):
        log["loai_bo"].append({"url": u, "source": s, "title": t, "ly_do": "Trùng nội dung (SHA-1)"})
    C._step(report, "Trùng nội dung (SHA-1)", "Dữ liệu trùng", n, len(df_h),
            "Băm SHA-1 nội dung đã chuẩn hóa, giữ 1 bản mỗi mã băm")

    bai = F.extract_features(df_h)

    # Loc bai TNGT (hoc ban giam sat) + ngoai lai
    pred, p, y0, info = M.self_training(bai)
    tong_hop = (bai["so_chet"].fillna(0) > F.MAX_CHET_MOT_VU) | \
               (bai["so_bi_thuong"].fillna(0) > F.MAX_BI_THUONG_MOT_VU)
    pred_ml = pred.copy()
    pred = np.where(tong_hop.values, 0, pred)
    bai["la_tngt"], bai["p_tngt"] = pred, p
    bai["ly_do_loai"] = np.where(pred == 1, "", np.where(tong_hop, "số liệu tổng hợp", "không phải tin TNGT"))
    bai["phuong_phap_loc"] = "lọc tự động (học bán giám sát)"
    n_off = int((pred_ml == 0).sum())
    C._step(report, "Bài không phải tin TNGT", "Dữ liệu nhiễu", len(bai), len(bai) - n_off,
            "Lọc tự động bằng học bán giám sát (Self-training)")
    C._step(report, f"Số chết > {F.MAX_CHET_MOT_VU} hoặc bị thương > {F.MAX_BI_THUONG_MOT_VU}",
            "Ngoại lai", len(bai) - n_off, int(pred.sum()), "Coi là bài thống kê tổng hợp, loại khỏi tập vụ")

    # Gop bai cung vu - MinHash-LSH
    vu, acc = F.gop_vu(bai[bai["la_tngt"] == 1])
    C._step(report, "Nhiều bài cùng một vụ", "Dữ liệu trùng", int(pred.sum()), len(vu),
            "MinHash-LSH + khối băm (tỉnh, ngày), (tỉnh, số chết) + Union-Find")
    ks = dict(vu.attrs.get("kiem_tra_trung", {}))

    # Gia tri thieu muc vu
    vu, thieu = C.xu_ly_thieu_vu(vu)

    # Bien doi
    X, vu_ext, rep_bd, demo = TR.bien_doi(vu)

    # Gan attrs cho app (chi kieu don gian de khong vuong pandas)
    bai.attrs["lam_sach"] = report
    bai.attrs["trung_hash"] = {"so_bai_goc": n_raw, "trung_noi_dung_sha1": n_hash,
                               "nhom_trung": grp.to_dict("records")}
    vu.attrs.update({"kiem_tra_trung": ks, "thieu": thieu.to_dict("records"), "lam_sach": report,
                     "vi_du_noi_dung": log["vi_du_noi_dung"], "vi_du_sapo": log["vi_du_sapo"]})
    bi_loai = pd.concat([pd.DataFrame(log["loai_bo"], columns=["url", "source", "title", "ly_do"]),
                         bai.loc[bai["la_tngt"] == 0, ["url", "source", "title", "ly_do_loai"]]
                         .rename(columns={"ly_do_loai": "ly_do"})], ignore_index=True)
    return {"bai": bai, "vu": vu, "acc": acc, "X": X, "bao_cao": pd.DataFrame(report),
            "thieu": thieu, "bien_doi": rep_bd, "demo": demo, "bi_loai": bi_loai}


def xuat(res, out_dir="data/processed"):
    """Ghi ket qua ra CSV (UTF-8 co BOM de mo bang Excel). Tra ve danh sach file."""
    os.makedirs(out_dir, exist_ok=True)
    files = {
        "01_bai_da_lam_sach.csv": res["bai"],
        "02_bai_bi_loai_va_ly_do.csv": res["bi_loai"],
        "03_vu_tai_nan_sach.csv": res["vu"].drop(columns=["content"], errors="ignore"),
        "04_bang_dac_trung.csv": res["X"],
        "05_bao_cao_lam_sach.csv": res["bao_cao"],
        "06_gia_tri_thieu.csv": res["thieu"],
        "07_bao_cao_bien_doi.csv": res["bien_doi"],
    }
    out = []
    for name, d in files.items():
        path = os.path.join(out_dir, name)
        d.to_csv(path, index=False, encoding="utf-8-sig")
        out.append((path, len(d), d.shape[1]))
    return out


if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else "data/tngt_raw.sqlite3"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(db) or ".", "processed")
    print(f"Doc {db} ...")
    res = chay(doc_sqlite(db))
    pd.set_option("display.width", 200, "display.max_colwidth", 60)
    print(res["bao_cao"][["Bước", "Loại vấn đề", "Trước", "Sau", "Thay đổi"]].to_string(index=False))
    for path, n, k in xuat(res, out_dir):
        print(f"  -> {path}: {n:,} dong x {k} cot")
