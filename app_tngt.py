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

import tngt_features as F

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
# Doc du lieu + trich xuat (co cache)
# ---------------------------------------------------------------------
@st.cache_data(show_spinner="Đang đọc và trích xuất dữ liệu...")
def load_from_sqlite(path, mtime):
    uri = "file:" + os.path.abspath(path).replace("\\", "/") + "?mode=ro"
    con = sqlite3.connect(uri, uri=True, timeout=30)
    try:
        df = pd.read_sql_query(
            "SELECT source, url, published, title, sapo, content FROM articles", con)
    finally:
        con.close()
    return prepare(df)


@st.cache_data(show_spinner="Đang đọc và trích xuất dữ liệu...")
def load_from_csv(raw_bytes, digest):
    df = pd.read_csv(io.BytesIO(raw_bytes), encoding="utf-8-sig")
    return prepare(df)


def prepare(df):
    for c in ("source", "url", "published", "title", "sapo", "content"):
        if c not in df:
            df[c] = None
    df = df.drop_duplicates("url")
    bai = F.extract_features(df)
    vu, bai_tngt = F.gop_vu(bai[bai["la_tngt"] == 1])
    return bai, vu


# ---------------------------------------------------------------------
# Sidebar: nguon du lieu + bo loc
# ---------------------------------------------------------------------
st.sidebar.title("🚦 TNGT đường bộ")
src_mode = st.sidebar.radio("Nguồn dữ liệu", ["File SQLite của crawler", "Tải lên CSV"], horizontal=False)
if src_mode == "File SQLite của crawler":
    path = st.sidebar.text_input("Đường dẫn", "data/tngt_raw.sqlite3")
    if not os.path.exists(path):
        st.info(f"Không tìm thấy `{os.path.abspath(path)}`. Hãy chạy app trong thư mục chứa "
                "thư mục `data/` của crawler, hoặc sửa đường dẫn / tải lên CSV ở thanh bên.")
        st.stop()
    bai, vu = load_from_sqlite(path, os.path.getmtime(path))
    if st.sidebar.button("🔄 Nạp lại dữ liệu mới"):
        st.cache_data.clear()
        st.rerun()
else:
    up = st.sidebar.file_uploader("File raw_articles.csv (lệnh export)", type="csv")
    if not up:
        st.info("Tải lên file CSV xuất từ crawler (`python crawl_tngt.py export`).")
        st.stop()
    raw = up.getvalue()
    bai, vu = load_from_csv(raw, hashlib.md5(raw).hexdigest())

if vu.empty:
    st.warning("Chưa có bài nào được nhận là tin tai nạn giao thông.")
    st.stop()

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


V = apply_filters(vu)
B = bai[bai["source"].isin(sel_src)]
if V.empty:
    st.warning("Không có vụ nào khớp bộ lọc.")
    st.stop()
base_rate = V["co_tu_vong"].mean()

st.title("Phân tích yếu tố gây thương vong trong TNGT đường bộ")
st.caption("Dữ liệu từ tin báo điện tử; các trường được trích xuất tự động theo luật nên là ước lượng, "
           "cần đối chiếu mẫu. Mỗi **vụ** có thể gộp nhiều bài báo cùng đưa tin.")

tabs = st.tabs(["📊 Tổng quan", "🕒 Thời gian", "📍 Địa điểm", "⚠️ Yếu tố & thương vong",
                "🔗 Luật kết hợp", "🗂️ Dữ liệu & chất lượng"])

# =====================================================================
# Tab 1: Tong quan
# =====================================================================
with tabs[0]:
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
with tabs[1]:
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
        known = V["khung_gio"].notna().mean()
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
with tabs[2]:
    p = V.dropna(subset=[prov_col]).groupby(prov_col).agg(
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
    st.caption(f"Xác định được tỉnh ở {V[prov_col].notna().mean():.0%} số vụ. Số vụ phản ánh mức độ "
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


with tabs[3]:
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
with tabs[4]:
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
with tabs[5]:
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
        fun = pd.DataFrame({"Bước": ["Bài đã cào", "Bài là tin TNGT", "Vụ sau gộp trùng"],
                            "Số lượng": [len(B), int(B["la_tngt"].sum()), len(V)]})
        st.dataframe(fun, hide_index=True)
        why = B.loc[B["la_tngt"] == 0, "ly_do_loai"].str.replace(r":.*", "", regex=True) \
            .value_counts().rename_axis("Lý do").reset_index(name="Số bài")
        st.dataframe(why, hide_index=True)
        cov = pd.DataFrame({"Trường": ["Tỉnh", "Ngày từ nội dung bài", "Khung giờ", "Số chết/bị thương",
                                       "Loại đường"],
                            "Tỷ lệ xác định được": [V[prov_col].notna().mean(),
                                                    (V["nguon_ngay"] != "ngay_dang").mean(),
                                                    V["khung_gio"].notna().mean(),
                                                    (V["so_chet"].notna() | V["so_bi_thuong"].notna()).mean(),
                                                    (V["loai_duong"] != "khác/không rõ").mean()]})
        st.dataframe(cov, hide_index=True,
                     column_config={"Tỷ lệ xác định được": st.column_config.NumberColumn(format="%.2f")})
        st.caption("Gợi ý cho báo cáo: lấy ngẫu nhiên ~100 vụ, tự đối chiếu với bài gốc để tính độ "
                   "chính xác từng trường.")