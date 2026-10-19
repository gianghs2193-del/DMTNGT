# -*- coding: utf-8 -*-
"""
app_tngt.py - App Streamlit hien thi ket qua bo du lieu TNGT duong bo.

Cai dat:  python -m pip install streamlit plotly pandas
Chay:     streamlit run app_tngt.py
          (dat file nay va tngt_features.py cung thu muc voi thu muc data/ cua crawler)

Doc truc tiep data/tngt_raw.sqlite3 (che do chi doc, chay duoc ca khi crawler dang cao)
hoac file CSV xuat tu lenh `export`.
"""
import hashlib
import io
import os
import sqlite3

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import tngt_clean as C
import tngt_features as F
import tngt_ml as M
import tngt_pipeline as P

st.set_page_config(page_title="TNGT đường bộ Việt Nam", page_icon="🚦", layout="wide")

# ---------------------------------------------------------------------
# Bang mau (palette da kiem dinh, sang/toi) - mau theo thuc the, thu tu co dinh
# ---------------------------------------------------------------------
LIGHT = {"cat": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
         "seq": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
         "muted": "#52514e", "grid": "rgba(0,0,0,0.08)"}
DARK = {"cat": ["#3987e5", "#d95926", "#199e70", "#c98500"],
        "seq": ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
        "muted": "#c3c2b7", "grid": "rgba(255,255,255,0.10)"}


def pal():
    try:
        return DARK if st.context.theme.type == "dark" else LIGHT
    except Exception:
        return LIGHT


def style(fig, height=380, legend=False):
    P = pal()
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), showlegend=legend,
                      bargap=0.35, hoverlabel=dict(font_size=13),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None))
    try:
        fig.update_layout(barcornerradius=4)
    except Exception:
        pass
    fig.update_xaxes(gridcolor=P["grid"], zeroline=False, title_font_color=P["muted"])
    fig.update_yaxes(gridcolor=P["grid"], zeroline=False, title_font_color=P["muted"])
    return fig


def show(fig, **kw):
    st.plotly_chart(style(fig, **kw), use_container_width=True, theme="streamlit")


# ---------------------------------------------------------------------
# Chay toan bo quy trinh (tngt_pipeline). Khoa cache gom PHIEN BAN MA NGUON: sua bat ky file
# xu ly nao thi app tu tinh lai, khong dung ket qua cu.
# ---------------------------------------------------------------------
@st.cache_data(show_spinner="Đang làm sạch, lọc, kiểm tra trùng bằng hàm băm và biến đổi dữ liệu...")
def run_all(src, version):
    df = P.doc_sqlite(src[1]) if src[0] == "sqlite" else pd.read_csv(io.BytesIO(src[1]), encoding="utf-8-sig")
    return P.chay(df)


def run_pipeline(src):
    r = run_all(src, P.code_version())
    return r["bai"], r["vu"], r["acc"]


def run_transform(src):
    r = run_all(src, P.code_version())
    return r["X"], r["bien_doi"], r["demo"]


# ---------------------------------------------------------------------
# Sidebar: nguon du lieu + phuong phap loc + bo loc
# ---------------------------------------------------------------------
st.sidebar.title("🚦 TNGT đường bộ")
src_mode = st.sidebar.radio("Nguồn dữ liệu", ["File SQLite của crawler", "Tải lên CSV"], horizontal=False)
if src_mode == "File SQLite của crawler":
    path = st.sidebar.text_input("Đường dẫn SQLite", "data/tngt_raw.sqlite3")
    if not os.path.exists(path):
        st.info(f"Không tìm thấy `{os.path.abspath(path)}`. Hãy chạy app trong thư mục chứa "
                "thư mục `data/` của crawler, hoặc sửa đường dẫn / tải lên CSV ở thanh bên.")
        st.stop()
    SRC = ("sqlite", path, os.path.getmtime(path))
    if st.sidebar.button("🔄 Nạp lại dữ liệu mới"):
        st.cache_data.clear()
        st.rerun()
else:
    up = st.sidebar.file_uploader("File raw_articles.csv (lệnh export)", type="csv")
    if not up:
        st.info("Tải lên file CSV xuất từ crawler (`python crawl_tngt.py export`).")
        st.stop()
    SRC = ("csv", up.getvalue())

bai, vu, ACC = run_pipeline(SRC)

if vu.empty:
    st.warning("Chưa có bài nào được nhận là tin tai nạn giao thông.")

st.sidebar.divider()
st.sidebar.subheader("Bộ lọc")
sources = sorted(bai["source"].dropna().unique())
sel_src = st.sidebar.multiselect("Báo", sources, default=sources)
dmin, dmax = vu["ngay_xay_ra"].min(), vu["ngay_xay_ra"].max()
if pd.isna(dmin):
    dmin = dmax = pd.Timestamp.today()
date_rng = st.sidebar.date_input("Ngày xảy ra", (dmin.date(), dmax.date()),
                                 min_value=dmin.date(), max_value=dmax.date())
prov_col = "tinh_moi" if st.sidebar.radio(
    "Đơn vị hành chính", ["34 tỉnh (sau 1/7/2025)", "63 tỉnh (cũ)"]) .startswith("34") else "tinh_cu"
provs = sorted(vu[prov_col].dropna().unique())
sel_prov = st.sidebar.multiselect("Tỉnh/thành", provs, placeholder="Tất cả")
min_n = st.sidebar.slider("Số vụ tối thiểu để so sánh tỷ lệ", 5, 50, 10,
                          help="Nhóm có ít vụ hơn mức này không được vẽ ở các biểu đồ tỷ lệ (tránh nhiễu).")


def apply_filters(d):
    m = d["cac_nguon"].map(lambda s: any(x in sel_src for x in str(s).split(", "))) \
        if "cac_nguon" in d else d["source"].isin(sel_src)
    if isinstance(date_rng, (list, tuple)) and len(date_rng) == 2:
        lo, hi = pd.Timestamp(date_rng[0]), pd.Timestamp(date_rng[1])
        m &= d["ngay_xay_ra"].between(lo, hi) | d["ngay_xay_ra"].isna()
    if sel_prov:
        m &= d[prov_col].isin(sel_prov)
    return d[m]


V = apply_filters(vu) if not vu.empty else vu
B = bai[bai["source"].isin(sel_src)]
base_rate = V["co_tu_vong"].mean() if not V.empty else np.nan

st.title("Phân tích yếu tố gây thương vong trong TNGT đường bộ")
st.caption("Trùng lặp được kiểm tra bằng **hàm băm** (SHA-1 + MinHash-LSH). Bài TNGT được lọc tự động. "
           "Các trường (tỉnh, thương vong, phương tiện...) trích xuất tự động theo luật nên là ước lượng, "
           "cần đối chiếu mẫu. Mỗi **vụ** có thể gộp nhiều bài báo cùng đưa tin.")

tabs = st.tabs(["🔑 Hàm băm & trùng lặp", "🧹 Làm sạch dữ liệu", "🔧 Biến đổi dữ liệu", "📊 Tổng quan", "🕒 Thời gian", "📍 Địa điểm",
                "⚠️ Yếu tố & thương vong", "🔗 Luật kết hợp", "🗂️ Dữ liệu & chất lượng"])

# =====================================================================
# Tab 1: Ham bam & trung lap (trong tam cua app; ve truoc de luon hien)
# =====================================================================
with tabs[0]:
    TH = bai.attrs.get("trung_hash", {})
    KS = vu.attrs.get("kiem_tra_trung", {})
    st.markdown("Hàm băm là **kỹ thuật chính** để kiểm tra trùng lặp, áp dụng ở 3 mức, từ trùng tuyệt đối "
                "đến trùng gần đúng. Mỗi phép kiểm tra trùng chỉ cần tra bảng băm, không phải so sánh "
                "từng cặp bài.")
    c = st.columns(5)
    c[0].metric("URL duy nhất", f"{TH.get('so_bai_goc', len(bai)):,}")
    c[1].metric("Trùng nội dung (SHA-1)", f"{TH.get('trung_noi_dung_sha1', 0):,}")
    c[2].metric("Bài tin TNGT", f"{int(bai['la_tngt'].sum()):,}", help="Lọc tự động trước khi gộp vụ")
    c[3].metric("Cặp ứng viên từ băm", f"{KS.get('cap_ung_vien_tong', 0):,}")
    c[4].metric("Số vụ sau gộp", f"{KS.get('so_vu', len(vu)):,}")

    # ---- Muc 1 ----
    st.subheader("Mức 1 · Băm URL: kiểm tra hàng đợi của crawler")
    st.markdown(
        "URL được chuẩn hóa (bỏ `?utm=...`, `#comment`) rồi băm **SHA-1**. Crawler giữ một bảng băm (`set`) "
        "chứa mã băm của mọi URL đã từng vào hàng đợi. Link mới chỉ được thêm khi mã băm **chưa có** trong "
        "bảng, nên mỗi lần kiểm tra mất O(1), không phải tra cơ sở dữ liệu.")

    # ---- Muc 2 ----
    st.subheader("Mức 2 · SHA-1 nội dung: bài trùng tuyệt đối")
    st.markdown(
        "Tiêu đề + nội dung được chuẩn hóa (chữ thường, bỏ dấu câu, gộp khoảng trắng) rồi băm SHA-1 thành "
        "chuỗi 40 ký tự. Hai bài giống hệt nhau (kể cả đăng ở hai báo khác nhau) cho cùng một mã băm. Bảng "
        "băm `mã băm → bài đầu tiên` giữ lại một bản, độ phức tạp O(n).")
    demo_a = "Lật xe chở sắt thép, hai bố con tử vong"
    demo_b = "Lật xe chở sắt thép, hai bố con tử vong."
    demo_c = "Lật xe chở sắt thép, hai bố con tử vọng"
    demo = pd.DataFrame({"Chuỗi đầu vào": [demo_a, demo_b, demo_c],
                         "Sau chuẩn hóa": [F.normalize_for_hash(x) for x in (demo_a, demo_b, demo_c)],
                         "SHA-1": [F.content_hash(x, "") for x in (demo_a, demo_b, demo_c)]})
    st.dataframe(demo, hide_index=True, use_container_width=True)
    st.caption("Dòng 1 và 2 chỉ khác dấu chấm nên sau chuẩn hóa cho cùng mã băm. Dòng 3 khác một dấu thanh "
               "nhưng mã băm khác hoàn toàn (hiệu ứng thác lũ). Vì vậy SHA-1 chỉ bắt được trùng tuyệt đối, "
               "cần thêm mức 3.")
    groups = pd.DataFrame(TH.get("nhom_trung", []))
    if groups.empty:
        st.info("Không có bài trùng tuyệt đối nội dung.")
    else:
        groups["ma_bam"] = groups["ma_bam"].str[:12] + "…"
        st.markdown(f"**Các nhóm bài trùng tuyệt đối tìm được** ({TH.get('trung_noi_dung_sha1', 0)} bài bị loại)")
        st.dataframe(groups.rename(columns={"ma_bam": "Mã băm", "source": "Báo", "title": "Tiêu đề"}),
                     hide_index=True, use_container_width=True,
                     column_config={"url": st.column_config.LinkColumn("Link")})

    # ---- Muc 3 ----
    st.subheader("Mức 3 · MinHash + LSH: nhiều báo đưa tin cùng một vụ")
    if KS:
        b_, r_ = KS["so_bang"], KS["so_hang_moi_bang"]
        st.markdown(
            f"1. **Shingling:** tách tiêu đề + sapo thành các cụm 2 từ (\"lật xe\", \"xe chở\", …).\n"
            f"2. **MinHash:** dùng {KS['so_ham_bam']} hàm băm hᵢ(x) = (aᵢ·x + bᵢ) mod p. Với mỗi hàm, lấy giá "
            "trị nhỏ nhất trên tập cụm, được chữ ký 128 số. Xác suất hai chữ ký trùng ở một vị trí "
            "**bằng đúng độ tương đồng Jaccard** của hai bài.\n"
            f"3. **LSH:** chia chữ ký thành {b_} băng × {r_} hàng, băm từng băng vào thùng. Hai bài rơi chung "
            "thùng ở ít nhất một băng thì thành cặp ứng viên. Thêm khối băm theo khóa (tỉnh, ngày) để bắt "
            "bài viết rất khác nhưng cùng tỉnh, cùng ngày, cùng số người chết.\n"
            "4. Cặp ứng viên cách nhau tối đa 1 ngày, cùng tỉnh và Jaccard ≥ 0,22 thì được gộp. "
            "Union-Find gom các cặp thành từng vụ.")
        c1, c2 = st.columns(2)
        with c1:
            s_ = np.linspace(0, 1, 101)
            curve = pd.DataFrame({"Jaccard": s_, "P": 1 - (1 - s_ ** r_) ** b_})
            fig = px.line(curve, x="Jaccard", y="P",
                          title=f"Xác suất thành cặp ứng viên (b={b_}, r={r_})",
                          labels={"Jaccard": "Độ tương đồng Jaccard thật", "P": "P(chung thùng băm)"})
            fig.update_traces(line=dict(width=2, color=pal()["cat"][0]),
                              hovertemplate="Jaccard %{x:.2f}: <b>%{y:.1%}</b><extra></extra>")
            fig.add_vline(x=0.22, line_dash="dot", line_color=pal()["muted"],
                          annotation_text="ngưỡng gộp 0,22", annotation_position="bottom right")
            fig.update_yaxes(tickformat=".0%")
            show(fig, height=340)
        with c2:
            cmp_ = pd.DataFrame({"Cách làm": ["So sánh vét cạn mọi cặp", "Ứng viên từ hàm băm"],
                                 "Số cặp phải so": [KS["cap_so_sanh_neu_vet_can"], KS["cap_ung_vien_tong"]]})
            fig = px.bar(cmp_, x="Cách làm", y="Số cặp phải so", log_y=True,
                         title=f"Số cặp phải so sánh: còn "
                               f"{KS['cap_ung_vien_tong'] / max(KS['cap_so_sanh_neu_vet_can'], 1):.1%}",
                         labels={"Cách làm": ""})
            fig.update_traces(marker_color=pal()["cat"][0], texttemplate="%{y:,}", textposition="outside",
                              hovertemplate="%{x}: <b>%{y:,} cặp</b><extra></extra>")
            show(fig, height=340)
        st.caption(f"Ngưỡng LSH xấp xỉ (1/b)^(1/r) = {KS['nguong_lsh_xap_xi']}: cặp có Jaccard trên mức này gần như "
                   "chắc chắn thành ứng viên. Kiểm tra trên dữ liệu: LSH bắt được khoảng 99% số cặp có "
                   "Jaccard ≥ 0,22, và số vụ ra trùng khớp với cách so sánh vét cạn.")

    multi = ACC[ACC.groupby("vu_id")["url"].transform("size") > 1] if len(ACC) else ACC
    if len(multi):
        st.markdown(f"**Ví dụ các vụ được gộp từ nhiều bài** ({multi['vu_id'].nunique():,} vụ có từ 2 bài trở lên)")
        top_ids = multi.groupby("vu_id")["url"].size().sort_values(ascending=False).index[:40]
        ex = multi[multi["vu_id"].isin(top_ids)].sort_values(["vu_id", "ngay_xay_ra"])
        ex = ex.assign(Vụ=ex["vu_id"].map({v: i + 1 for i, v in enumerate(top_ids)})).sort_values("Vụ")
        st.dataframe(ex[["Vụ", "ngay_xay_ra", "source", "title", "url"]], hide_index=True,
                     use_container_width=True, height=380,
                     column_config={"url": st.column_config.LinkColumn("Link"), "source": "Báo",
                                    "title": "Tiêu đề", "ngay_xay_ra": st.column_config.DateColumn(
                                        "Ngày", format="DD/MM/YYYY")})
    st.caption("Trước khi gộp vụ, bài được lọc tự động để chỉ giữ tin TNGT đường bộ (bước tiền xử lý cố định "
               "bằng học bán giám sát, không gán nhãn tay).")

# =====================================================================
# Tab 2: Lam sach du lieu
# =====================================================================
with tabs[1]:
    rp = pd.DataFrame(vu.attrs.get("lam_sach", []))
    st.markdown("Các bước làm sạch chạy tự động mỗi lần nạp dữ liệu, theo thứ tự dưới đây. Mỗi bước ghi "
                "số bản ghi trước và sau khi xử lý.")
    if not rp.empty:
        c = st.columns(4)
        for col, loai in zip(c, ["Dữ liệu trùng", "Giá trị thiếu", "Dữ liệu nhiễu", "Ngoại lai"]):
            col.metric(loai, f"{-int(rp.loc[rp['Loại vấn đề'] == loai, 'Thay đổi'].sum()):,} bị loại")
        st.dataframe(rp, hide_index=True, use_container_width=True)
        g = rp.groupby("Loại vấn đề", sort=False)["Thay đổi"].sum().abs().reset_index()
        fig = px.bar(g.sort_values("Thay đổi"), y="Loại vấn đề", x="Thay đổi", orientation="h",
                     title="Số bản ghi bị loại/gộp theo loại vấn đề", labels={"Thay đổi": "Số bản ghi", "Loại vấn đề": ""})
        fig.update_traces(marker_color=pal()["cat"][0], texttemplate="%{x:,}", textposition="outside",
                          hovertemplate="%{y}: <b>%{x:,}</b><extra></extra>")
        show(fig, height=280)
    th = pd.DataFrame(vu.attrs.get("thieu", []))
    if not th.empty:
        st.subheader("Giá trị thiếu sau khi trích xuất (trên bảng vụ)")
        st.dataframe(th, hide_index=True, use_container_width=True, column_config={
            "% thiếu trước": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=1),
            "% 'Không rõ' sau": st.column_config.NumberColumn(format="%.2f")})
        st.caption("Không suy đoán (impute) tỉnh, giờ hay loại đường, vì điền sai sẽ làm lệch kết quả phân "
                   "tích. Các giá trị này được gắn nhãn \"Không rõ\" và tách riêng khi vẽ biểu đồ. Số người chết "
                   "hoặc bị thương không được nhắc trong bài thì điền 0 kèm cột cờ để vẫn phân biệt được.")
    st.subheader("Ví dụ trước / sau khi làm sạch")
    ex1 = pd.DataFrame(vu.attrs.get("vi_du_noi_dung", []))
    if not ex1.empty:
        st.markdown("**Dòng rác đã xóa khỏi nội dung bài**")
        st.dataframe(ex1.rename(columns={"title": "Bài", "da_xoa": "Đã xóa"}), hide_index=True,
                     use_container_width=True)
    ex2 = pd.DataFrame(vu.attrs.get("vi_du_sapo", []))
    if not ex2.empty:
        st.markdown("**Sapo bị thiếu → điền bằng câu đầu bài**")
        st.dataframe(ex2.rename(columns={"title": "Bài", "sapo_cu": "Sapo cũ", "sapo_moi": "Sapo mới"}),
                     hide_index=True, use_container_width=True)
    ex3 = vu[["title", "so_chet", "thieu_so_chet", "so_bi_thuong", "thieu_so_bi_thuong", "tinh_moi",
              "khung_gio", "loai_duong"]]
    ex3 = ex3[(ex3["thieu_so_chet"] == 1) | (ex3["tinh_moi"] == "Không rõ") | (ex3["khung_gio"].astype(str) == "Không rõ")]
    if len(ex3):
        st.markdown("**Vụ có giá trị thiếu đã được xử lý** (0 kèm cờ = 1 là giá trị điền vào; \"Không rõ\" là nhãn)")
        st.dataframe(ex3.head(15), hide_index=True, use_container_width=True)
    BL = run_all(SRC, P.code_version())["bi_loai"]
    with st.expander(f"Danh sách {len(BL):,} bài bị loại và lý do"):
        st.dataframe(BL["ly_do"].value_counts().rename_axis("Lý do").reset_index(name="Số bài"), hide_index=True)
        st.dataframe(BL, hide_index=True, use_container_width=True, height=300,
                     column_config={"url": st.column_config.LinkColumn("Link")})

    st.subheader("Lưu dữ liệu đã xử lý ra file")
    st.caption("Dữ liệu thô trong SQLite được giữ nguyên. Kết quả làm sạch và biến đổi được ghi thành các "
               "file CSV riêng, mở được bằng Excel.")
    out_dir = os.path.join(os.path.dirname(SRC[1]) if SRC[0] == "sqlite" else "data", "processed")
    if st.button(f"💾 Lưu vào {out_dir}"):
        for path_, n_, k_ in P.xuat(run_all(SRC, P.code_version()), out_dir):
            st.write(f"✅ `{path_}`: {n_:,} dòng × {k_} cột")

    st.subheader("Ngoại lai")
    st.markdown(f"- Một vụ có trên **{F.MAX_CHET_MOT_VU} người chết** hoặc trên **{F.MAX_BI_THUONG_MOT_VU} người "
                "bị thương** được coi là bài thống kê tổng hợp và bị loại, vì vụ TNGT đường bộ nghiêm trọng nhất "
                "ở Việt Nam hiếm khi vượt mức này.\n"
                "- Khi đếm thương vong, chỉ tính con số đi kèm danh từ chỉ người (\"3 người\", \"hai bố con\"), "
                "để tránh đếm nhầm \"lớp 12\", \"29 Tết\", \"20 con heo\", \"thoát chết\".")
    if len(vu):
        top = vu.nlargest(10, "so_chet")[["so_chet", "so_bi_thuong", "tinh_moi", "title", "url"]]
        st.markdown("**10 vụ có số người chết cao nhất** (nên đọc lại để kiểm tra ngoại lai còn sót)")
        st.dataframe(top, hide_index=True, use_container_width=True,
                     column_config={"url": st.column_config.LinkColumn("Link")})

# =====================================================================
# Tab 3: Bien doi du lieu (rut trich dac trung, chuan hoa, ma hoa)
# =====================================================================
with tabs[2]:
    if len(vu):
        X, TREP, TDEMO = run_transform(SRC)
        st.markdown("Bảng **vụ** sau làm sạch được biến đổi thành **bảng đặc trưng toàn số**, sẵn sàng cho "
                    "luật kết hợp, phân cụm và phân lớp. Thứ tự: rút trích đặc trưng → chuẩn hóa → mã hóa.")
        c = st.columns(3)
        c[0].metric("Số vụ (dòng)", f"{len(X):,}")
        c[1].metric("Số đặc trưng (cột)", f"{X.shape[1] - 2:,}", help="Không tính vu_id và cột nhãn")
        c[2].metric("Nhãn: vụ có người chết", f"{X['NHAN_co_tu_vong'].mean():.0%}")
        st.dataframe(TREP, hide_index=True, use_container_width=True)

        st.subheader("Ví dụ chuẩn hóa: số người chết")
        desc = TDEMO.describe().T[["min", "max", "mean", "std"]].rename(
            columns={"min": "Nhỏ nhất", "max": "Lớn nhất", "mean": "Trung bình", "std": "Độ lệch chuẩn"})
        st.dataframe(desc, use_container_width=True,
                     column_config={k: st.column_config.NumberColumn(format="%.3f") for k in desc.columns})
        st.caption("log(1+x) giảm ảnh hưởng của các vụ rất nặng; Min-Max đưa về [0, 1]; Z-score đưa về "
                   "trung bình 0, độ lệch chuẩn 1, để các đặc trưng có thang đo ngang nhau.")

        st.subheader("Đặc trưng liên quan nhiều nhất tới tử vong")
        bin_cols = [c_ for c_ in X.columns if X[c_].dropna().isin([0, 1]).all()
                    and c_ not in ("NHAN_co_tu_vong", "vu_id") and not c_.startswith(("chet_", "thieu_"))
                    and X[c_].sum() >= 10]
        corr = X[bin_cols].corrwith(X["NHAN_co_tu_vong"]).dropna()
        top = pd.concat([corr.nlargest(8), corr.nsmallest(7)]).sort_values().reset_index()
        top.columns = ["Đặc trưng", "Tương quan"]
        fig = px.bar(top, y="Đặc trưng", x="Tương quan", orientation="h",
                     title="Hệ số tương quan (phi) của đặc trưng nhị phân với nhãn có tử vong",
                     labels={"Đặc trưng": ""})
        fig.update_traces(marker_color=[pal()["cat"][1] if v > 0 else pal()["cat"][0] for v in top["Tương quan"]],
                          hovertemplate="%{y}: <b>%{x:.3f}</b><extra></extra>")
        fig.add_vline(x=0, line_color=pal()["muted"])
        show(fig, height=460)
        st.caption("Dương: đặc trưng đi kèm vụ có người chết nhiều hơn; âm: ít hơn. Đây là tương quan, "
                   "không phải nhân quả.")

        st.subheader("Xem trước bảng đặc trưng")
        st.dataframe(X.head(30), use_container_width=True, height=300)
        st.download_button("⬇️ Tải bảng đặc trưng (CSV)", X.to_csv(index=False).encode("utf-8-sig"),
                           "dac_trung_tngt.csv", "text/csv")

if V.empty:
    for t in tabs[3:]:
        with t:
            st.info("Chưa có vụ nào khớp bộ lọc / chưa có bài được nhận là TNGT.")
    st.stop()

# =====================================================================
# Tab 1: Tong quan
# =====================================================================
with tabs[3]:
    c = st.columns(5)
    c[0].metric("Bài báo đã cào", f"{len(B):,}")
    c[1].metric("Bài là tin TNGT", f"{int(B['la_tngt'].sum()):,}")
    c[2].metric("Số vụ (sau gộp trùng)", f"{len(V):,}")
    c[3].metric("Người chết (ước tính)", f"{int(V['so_chet'].fillna(0).sum()):,}")
    c[4].metric("Người bị thương (ước tính)", f"{int(V['so_bi_thuong'].fillna(0).sum()):,}")

    left, right = st.columns([2, 1])
    with left:
        m = V.dropna(subset=["thang"]).groupby("thang").agg(
            so_vu=("vu_id", "size"), so_chet=("so_chet", "sum")).reset_index()
        fig = px.line(m, x="thang", y="so_vu", markers=True,
                      labels={"thang": "Tháng", "so_vu": "Số vụ"}, title="Số vụ theo tháng")
        fig.update_traces(line=dict(width=2, color=pal()["cat"][0]), marker=dict(size=8),
                          hovertemplate="%{x|%m/%Y}: <b>%{y} vụ</b><extra></extra>")
        fig.update_layout(hovermode="x unified")
        show(fig)
    with right:
        s = B.groupby("source").agg(bai=("url", "size"), tngt=("la_tngt", "sum")).reset_index()
        s["tỷ lệ TNGT"] = s["tngt"] / s["bai"]
        s = s.sort_values("tngt")
        fig = px.bar(s, y="source", x="tngt", orientation="h", title="Bài TNGT theo báo",
                     labels={"source": "", "tngt": "Số bài TNGT"},
                     custom_data=["bai", "tỷ lệ TNGT"])
        fig.update_traces(marker_color=pal()["cat"][0], texttemplate="%{x:,}", textposition="outside",
                          hovertemplate="%{y}: %{x:,} / %{customdata[0]:,} bài (%{customdata[1]:.0%})"
                                        "<extra></extra>")
        show(fig)

    sev = V["muc_do"].value_counts().reindex(
        ["Có người chết", "Chỉ bị thương", "Không rõ/không thương vong"]).fillna(0).reset_index()
    sev.columns = ["Mức độ", "Số vụ"]
    sev["%"] = sev["Số vụ"] / sev["Số vụ"].sum()
    st.markdown("**Mức độ nghiêm trọng**")
    sev["%"] = (sev["%"] * 100).round(1)
    st.dataframe(sev, hide_index=True,
                 column_config={"%": st.column_config.ProgressColumn("% số vụ", format="%.1f%%",
                                                                     min_value=0, max_value=100)})

# =====================================================================
# Tab 2: Thoi gian
# =====================================================================
with tabs[4]:
    c1, c2 = st.columns(2)
    with c1:
        w = V["thu"].value_counts().reindex(F.WEEKDAYS).fillna(0).reset_index()
        w.columns = ["Thứ", "Số vụ"]
        fig = px.bar(w, x="Thứ", y="Số vụ", title="Số vụ theo thứ trong tuần")
        fig.update_traces(marker_color=pal()["cat"][0], hovertemplate="%{x}: <b>%{y} vụ</b><extra></extra>")
        show(fig)
    with c2:
        t = V["khung_gio"].value_counts().reindex(F.TIME_BUCKETS).fillna(0).reset_index()
        t.columns = ["Khung giờ", "Số vụ"]
        known = (V["khung_gio"].astype(str) != "Không rõ").mean()
        fig = px.bar(t, x="Khung giờ", y="Số vụ",
                     title=f"Số vụ theo khung giờ (xác định được giờ ở {known:.0%} số vụ)")
        fig.update_traces(marker_color=pal()["cat"][0], hovertemplate="%{x}: <b>%{y} vụ</b><extra></extra>")
        show(fig)

    hm = pd.crosstab(V["khung_gio"], V["thu"]).reindex(index=F.TIME_BUCKETS, columns=F.WEEKDAYS).fillna(0)
    fig = px.imshow(hm, text_auto=True, aspect="auto", color_continuous_scale=pal()["seq"],
                    labels=dict(x="", y="", color="Số vụ"), title="Bản đồ nhiệt: khung giờ × thứ")
    fig.update_traces(hovertemplate="%{y}, %{x}: <b>%{z} vụ</b><extra></extra>", xgap=2, ygap=2)
    show(fig, height=330)

    c3, c4 = st.columns(2)
    with c3:
        g = V.groupby("khung_gio", observed=False).agg(n=("vu_id", "size"), r=("co_tu_vong", "mean")).reset_index()
        g = g[g["n"] >= min_n]
        fig = px.bar(g, x="khung_gio", y="r", title="Tỷ lệ vụ có người chết theo khung giờ",
                     labels={"khung_gio": "", "r": "% vụ có người chết"}, custom_data=["n"])
        fig.update_traces(marker_color=pal()["cat"][0],
                          hovertemplate="%{x}: <b>%{y:.0%}</b> (n=%{customdata[0]})<extra></extra>")
        fig.add_hline(y=base_rate, line_dash="dot", line_color=pal()["muted"],
                      annotation_text=f"Trung bình {base_rate:.0%}", annotation_position="top left")
        fig.update_yaxes(tickformat=".0%")
        show(fig)
    with c4:
        le = V.groupby("dip_le_tet").agg(n=("vu_id", "size"), r=("co_tu_vong", "mean"),
                                         chet=("so_chet", "mean")).reset_index()
        le["Thời điểm"] = le["dip_le_tet"].map({0: "Ngày thường", 1: "Dịp lễ/Tết (±7 ngày Tết, ±2 ngày lễ)"})
        st.markdown("**Ngày thường so với dịp lễ, Tết**")
        st.dataframe(le[["Thời điểm", "n", "r", "chet"]].rename(
            columns={"n": "Số vụ", "r": "% có người chết", "chet": "Số chết TB/vụ"}),
            hide_index=True, column_config={
                "% có người chết": st.column_config.NumberColumn(format="%.2f"),
                "Số chết TB/vụ": st.column_config.NumberColumn(format="%.2f")})
        st.caption("Số vụ theo tháng nằm ở tab Tổng quan. Ngày xảy ra lấy từ bài (\"tối 15/10\", "
                   "\"đêm qua\"), nếu không có thì dùng ngày đăng.")

# =====================================================================
# Tab 3: Dia diem
# =====================================================================
with tabs[5]:
    p = V[V[prov_col] != "Không rõ"].groupby(prov_col).agg(
        so_vu=("vu_id", "size"), so_chet=("so_chet", "sum"),
        ty_le_chet=("co_tu_vong", "mean")).reset_index()
    c1, c2 = st.columns(2)
    with c1:
        top = p.nlargest(15, "so_vu").sort_values("so_vu")
        fig = px.bar(top, y=prov_col, x="so_vu", orientation="h", title="15 tỉnh/thành có nhiều vụ nhất",
                     labels={prov_col: "", "so_vu": "Số vụ"}, custom_data=["so_chet"])
        fig.update_traces(marker_color=pal()["cat"][0],
                          hovertemplate="%{y}: <b>%{x} vụ</b>, %{customdata[0]:.0f} người chết<extra></extra>")
        show(fig, height=480)
    with c2:
        q = p[p["so_vu"] >= min_n].nlargest(15, "ty_le_chet").sort_values("ty_le_chet")
        fig = px.bar(q, y=prov_col, x="ty_le_chet", orientation="h",
                     title=f"Tỷ lệ vụ có người chết (tỉnh có ≥{min_n} vụ)",
                     labels={prov_col: "", "ty_le_chet": "% vụ có người chết"}, custom_data=["so_vu"])
        fig.update_traces(marker_color=pal()["cat"][1],
                          hovertemplate="%{y}: <b>%{x:.0%}</b> (n=%{customdata[0]})<extra></extra>")
        fig.add_vline(x=base_rate, line_dash="dot", line_color=pal()["muted"])
        fig.update_xaxes(tickformat=".0%")
        show(fig, height=480)
    st.caption(f"Xác định được tỉnh ở {(V[prov_col] != "Không rõ").mean():.0%} số vụ. Số vụ phản ánh mức độ "
               "được báo chí đưa tin, không phải số liệu chính thức: tỉnh có nhiều tin không nhất thiết "
               "có nhiều tai nạn hơn.")

    r = V.groupby("loai_duong").agg(n=("vu_id", "size"), r=("co_tu_vong", "mean"),
                                    chet=("so_chet", "mean")).reset_index().sort_values("n")
    c3, c4 = st.columns(2)
    with c3:
        fig = px.bar(r, y="loai_duong", x="n", orientation="h", title="Số vụ theo loại đường",
                     labels={"loai_duong": "", "n": "Số vụ"})
        fig.update_traces(marker_color=pal()["cat"][0], hovertemplate="%{y}: <b>%{x} vụ</b><extra></extra>")
        show(fig)
    with c4:
        rr = r[r["n"] >= min_n].sort_values("r")
        fig = px.bar(rr, y="loai_duong", x="r", orientation="h", title="Tỷ lệ vụ có người chết theo loại đường",
                     labels={"loai_duong": "", "r": "% vụ có người chết"}, custom_data=["n"])
        fig.update_traces(marker_color=pal()["cat"][1],
                          hovertemplate="%{y}: <b>%{x:.0%}</b> (n=%{customdata[0]})<extra></extra>")
        fig.add_vline(x=base_rate, line_dash="dot", line_color=pal()["muted"])
        fig.update_xaxes(tickformat=".0%")
        show(fig)
    with st.expander("Bảng chi tiết theo tỉnh"):
        st.dataframe(p.sort_values("so_vu", ascending=False).rename(columns={
            prov_col: "Tỉnh/thành", "so_vu": "Số vụ", "so_chet": "Số người chết",
            "ty_le_chet": "% vụ có người chết"}), hide_index=True, use_container_width=True,
            column_config={"% vụ có người chết": st.column_config.NumberColumn(format="%.2f")})

# =====================================================================
# Tab 4: Yeu to & thuong vong
# =====================================================================
GROUPS = {"Phương tiện liên quan": "pt_", "Nguyên nhân / hành vi": "nn_",
          "Kiểu tai nạn": "kieu_", "Thời tiết": "tt_"}


def factor_table(d, prefix):
    rows = []
    for c in [c for c in d.columns if c.startswith(prefix)]:
        sub = d[d[c] == 1]
        if len(sub) == 0:
            continue
        rows.append({"Yếu tố": c.split("_", 1)[1], "Số vụ": len(sub),
                     "% có người chết": sub["co_tu_vong"].mean(),
                     "Số chết TB/vụ": sub["so_chet"].fillna(0).mean(),
                     "Lift": sub["co_tu_vong"].mean() / base_rate if base_rate else np.nan})
    return pd.DataFrame(rows)


with tabs[6]:
    st.markdown(f"So sánh **tỷ lệ vụ có người chết** khi có từng yếu tố với mức trung bình "
                f"**{base_rate:.0%}**. Lift > 1 nghĩa là yếu tố đó đi kèm tử vong nhiều hơn bình thường "
                "(tương quan, chưa phải nhân quả).")
    grp = st.radio("Nhóm yếu tố", list(GROUPS), horizontal=True)
    ft = factor_table(V, GROUPS[grp])
    if ft.empty:
        st.info("Không có dữ liệu cho nhóm này.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            d = ft.sort_values("Số vụ")
            fig = px.bar(d, y="Yếu tố", x="Số vụ", orientation="h", title="Số vụ có yếu tố")
            fig.update_traces(marker_color=pal()["cat"][0], texttemplate="%{x}", textposition="outside",
                              hovertemplate="%{y}: <b>%{x} vụ</b><extra></extra>")
            show(fig, height=max(320, 40 * len(d) + 80))
        with c2:
            d = ft[ft["Số vụ"] >= min_n].sort_values("% có người chết")
            fig = px.bar(d, y="Yếu tố", x="% có người chết", orientation="h",
                         title=f"% vụ có người chết (yếu tố có ≥{min_n} vụ)",
                         custom_data=["Số vụ", "Lift"])
            fig.update_traces(marker_color=pal()["cat"][1],
                              hovertemplate="%{y}: <b>%{x:.0%}</b> (n=%{customdata[0]}, "
                                            "lift %{customdata[1]:.2f})<extra></extra>")
            fig.add_vline(x=base_rate, line_dash="dot", line_color=pal()["muted"],
                          annotation_text="trung bình", annotation_position="top")
            fig.update_xaxes(tickformat=".0%", range=[0, 1])
            show(fig, height=max(320, 40 * len(d) + 80))
        st.dataframe(ft.sort_values("Lift", ascending=False), hide_index=True, use_container_width=True,
                     column_config={"% có người chết": st.column_config.NumberColumn(format="%.2f"),
                                    "Số chết TB/vụ": st.column_config.NumberColumn(format="%.2f"),
                                    "Lift": st.column_config.NumberColumn(format="%.2f")})

    st.divider()
    st.markdown("**Kết hợp hai yếu tố**: % vụ có người chết theo phương tiện × nguyên nhân")
    pts = [c for c in V.columns if c.startswith("pt_")]
    nns = [c for c in V.columns if c.startswith("nn_")]
    grid = pd.DataFrame(index=[c[3:] for c in pts], columns=[c[3:] for c in nns], dtype=float)
    cnt = grid.copy()
    for a in pts:
        for b in nns:
            sub = V[(V[a] == 1) & (V[b] == 1)]
            cnt.loc[a[3:], b[3:]] = len(sub)
            grid.loc[a[3:], b[3:]] = sub["co_tu_vong"].mean() if len(sub) >= max(3, min_n // 2) else np.nan
    fig = px.imshow(grid, aspect="auto", color_continuous_scale=pal()["seq"], zmin=0, zmax=1,
                    labels=dict(color="% có chết"))
    fig.update_traces(customdata=cnt.values, xgap=2, ygap=2,
                      hovertemplate="%{y} + %{x}: <b>%{z:.0%}</b> (n=%{customdata:.0f})<extra></extra>")
    fig.update_coloraxes(colorbar_tickformat=".0%")
    show(fig, height=380)
    st.caption(f"Ô trống: tổ hợp có quá ít vụ (< {max(3, min_n // 2)}).")

# =====================================================================
# Tab 5: Luat ket hop
# =====================================================================
with tabs[7]:
    st.markdown("Tìm các tổ hợp yếu tố hay xuất hiện cùng nhau bằng **Apriori** "
                "(support = tỷ lệ vụ chứa tổ hợp, confidence = P(vế phải | vế trái), "
                "lift = confidence / tỷ lệ chung của vế phải).")
    c1, c2, c3, c4 = st.columns(4)
    sup = c1.slider("Support tối thiểu", 0.005, 0.2, 0.02, 0.005, format="%.3f")
    conf = c2.slider("Confidence tối thiểu", 0.1, 1.0, 0.5, 0.05)
    mlen = c3.selectbox("Số hạng mục tối đa", [2, 3, 4], index=1)
    only_death = c4.checkbox("Chỉ luật → CÓ TỬ VONG", value=True)
    items = F.build_items(V)
    rules = F.apriori_rules(items, sup, conf, mlen, target="CÓ TỬ VONG" if only_death else None)
    if rules.empty:
        st.info("Không có luật nào thỏa ngưỡng. Thử giảm support hoặc confidence.")
    else:
        st.caption(f"{len(rules)} luật, trên {len(items)} vụ.")
        st.dataframe(rules.head(200), hide_index=True, use_container_width=True,
                     column_config={"support": st.column_config.NumberColumn(format="%.3f"),
                                    "confidence": st.column_config.NumberColumn(format="%.2f"),
                                    "lift": st.column_config.NumberColumn(format="%.2f")})
        fig = px.scatter(rules.head(300), x="support", y="confidence", color="lift",
                         color_continuous_scale=pal()["seq"], hover_data=["Nếu (vế trái)", "Thì", "số vụ"],
                         title="Các luật: support × confidence (màu = lift)")
        fig.update_traces(marker=dict(size=10, line=dict(width=1, color="rgba(255,255,255,0.8)")))
        show(fig, height=420)
        st.download_button("⬇️ Tải bảng luật (CSV)", rules.to_csv(index=False).encode("utf-8-sig"),
                           "luat_ket_hop.csv", "text/csv")

# =====================================================================
# Tab 6: Du lieu & chat luong
# =====================================================================
with tabs[8]:
    show_cols = ["ngay_xay_ra", prov_col, "title", "so_chet", "so_bi_thuong", "muc_do", "khung_gio",
                 "loai_duong", "so_bai", "cac_nguon", "url"]
    q = st.text_input("Tìm trong tiêu đề", "")
    T = V[V["title"].str.contains(q, case=False, na=False)] if q else V
    st.dataframe(T[show_cols].sort_values("ngay_xay_ra", ascending=False), hide_index=True,
                 use_container_width=True, height=420,
                 column_config={"url": st.column_config.LinkColumn("Link"),
                                "ngay_xay_ra": st.column_config.DateColumn("Ngày", format="DD/MM/YYYY")})
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Tải bảng vụ đã trích xuất (CSV)",
                       V.drop(columns=["content"]).to_csv(index=False).encode("utf-8-sig"),
                       "tngt_vu.csv", "text/csv")
    c2.download_button("⬇️ Tải toàn bộ bài kèm nhãn (CSV)",
                       B.drop(columns=["content"]).to_csv(index=False).encode("utf-8-sig"),
                       "tngt_bai_co_nhan.csv", "text/csv")

    with st.expander("Xem chi tiết một vụ"):
        opts = T.sort_values("ngay_xay_ra", ascending=False).head(500)
        pick = st.selectbox("Chọn vụ", opts.index, format_func=lambda i: f"{opts.at[i, 'title']}")
        if pick is not None:
            r = opts.loc[pick]
            st.markdown(f"**{r['title']}**  \n{r['cac_nguon']} · {r['so_bai']} bài · [link]({r['url']})")
            st.write(r["sapo"])
            flags = [c.split("_", 1)[1] for c in V.columns
                     if c.startswith(("pt_", "nn_", "kieu_", "tt_")) and r[c] == 1]
            st.markdown(f"Trích xuất: tỉnh **{r[prov_col]}**, chết **{r['so_chet']}**, bị thương "
                        f"**{r['so_bi_thuong']}**, giờ **{r['khung_gio']}**, đường **{r['loai_duong']}**  \n"
                        f"Yếu tố: {', '.join(flags) or '—'}")
            st.text_area("Nội dung", r["content"] or "", height=250)

    with st.expander("Chất lượng dữ liệu: phễu lọc và lý do loại bài"):
        th = bai.attrs.get("trung_hash", {})
        fun = pd.DataFrame({"Bước": ["Bài đã cào (URL duy nhất)", "Loại bài trùng nội dung (SHA-1)",
                                     "Bài là tin TNGT", "Vụ sau gộp trùng (MinHash-LSH)"],
                            "Số lượng": [th.get("so_bai_goc", len(bai)), -th.get("trung_noi_dung_sha1", 0),
                                         int(B["la_tngt"].sum()), len(V)]})
        st.dataframe(fun, hide_index=True)
        why = B.loc[B["la_tngt"] == 0, "ly_do_loai"].str.replace(r":.*", "", regex=True) \
            .value_counts().rename_axis("Lý do").reset_index(name="Số bài")
        st.dataframe(why, hide_index=True)
        cov = pd.DataFrame({"Trường": ["Tỉnh", "Ngày từ nội dung bài", "Khung giờ", "Số chết/bị thương",
                                       "Loại đường"],
                            "Tỷ lệ xác định được": [(V[prov_col] != "Không rõ").mean(),
                                                    (V["nguon_ngay"] != "ngay_dang").mean(),
                                                    (V["khung_gio"].astype(str) != "Không rõ").mean(),
                                                    ((V["thieu_so_chet"] == 0) | (V["thieu_so_bi_thuong"] == 0)).mean(),
                                                    (V["loai_duong"] != "Không rõ").mean()]})
        st.dataframe(cov, hide_index=True,
                     column_config={"Tỷ lệ xác định được": st.column_config.NumberColumn(format="%.2f")})
        st.caption("Gợi ý cho báo cáo: lấy ngẫu nhiên ~100 vụ, tự đối chiếu với bài gốc để tính độ "
                   "chính xác từng trường.")
