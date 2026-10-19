# -*- coding: utf-8 -*-
"""
tngt_ml.py - Loc bai "la tin mot vu TNGT duong bo" bang THUAT TOAN HOC MAY,
KHONG can gan nhan tay: Hoc ban giam sat - Self-training.
   - Nhan yeu (weak labels) tu dong cho cac bai "chac chan":
       + duong:  luat nhan la TNGT, tieu de co hanh vi tai nan VA tim thay so thuong vong
       + am:     khong co tu khoa tai nan nao, hoac bi loai vi tu khoa loai tru /
                 tin nuoc ngoai / so lieu tong hop
     Cac bai con lai (mo ho) de trong nhan (-1).
   - SelfTrainingClassifier (scikit-learn) + Logistic Regression tren TF-IDF:
     mo hinh tu gan nhan dan cho bai mo ho co do tin cay >= nguong, lap den khi on dinh.
   - Du doan cuoi bang kiem dinh cheo (out-of-fold): moi bai duoc cham boi mo hinh
     KHONG hoc tren chinh no -> mo hinh co the "lat" nhan yeu sai cua luat (phat hien nhan nhieu).

Dung lam BUOC TIEN XU LY CO DINH trong app (khong con lua chon thuat toan); trong tam cua
app la kiem tra trung lap bang HAM BAM (xem tngt_features.py).
"""
import numpy as np
import pandas as pd

SEED = 42


def model_text(title, sapo, content):
    return f"{title or ''}. {sapo or ''}. {(content or '')[:500]}"


def texts_of(df):
    return [model_text(t, s, c) for t, s, c in zip(df["title"], df["sapo"], df["content"])]


def _tfidf(max_features=40000, min_df=3):
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, max_df=0.5, sublinear_tf=True,
                           max_features=max_features)


# =====================================================================
# 1. Hoc ban giam sat (Self-training)
# =====================================================================
def weak_labels(bai):
    """1 = duong chac chan, 0 = am chac chan, -1 = chua biet. Dua tren cac cot cua tngt_features."""
    import tngt_features as F
    title_acc = bai["title"].fillna("").map(lambda t: bool(F.ACCIDENT_RE.search(t)))
    has_cas = bai["so_chet"].notna() | bai["so_bi_thuong"].notna()
    pos = (bai["la_tngt_luat"] == 1) & title_acc & has_cas
    why = bai["ly_do_loai_luat"].fillna("")
    neg = (~bai["co_tu_khoa_tn"]) | why.str.startswith(("tu khoa loai tru", "tin nuoc ngoai", "so lieu"))
    y = np.full(len(bai), -1)
    y[neg.values] = 0
    y[pos.values] = 1
    return y


def self_training(bai, threshold=0.9):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.semi_supervised import SelfTrainingClassifier

    texts = texts_of(bai)
    vec = _tfidf()
    X = vec.fit_transform(texts)
    y0 = weak_labels(bai)

    base = LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000)
    st = SelfTrainingClassifier(base, threshold=threshold, max_iter=20)
    st.fit(X, y0)
    y1 = st.transduction_                       # nhan sau tu huan luyen (-1 = van chua gan)
    n_iter = int(st.n_iter_)
    pseudo = int(((y0 == -1) & (y1 != -1)).sum())

    # Du doan out-of-fold tren cac bai da co nhan -> phat hien nhan yeu sai
    lab = y1 != -1
    proba = np.full(len(bai), np.nan)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    proba[lab] = cross_val_predict(
        LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000),
        X[lab], y1[lab], cv=cv, method="predict_proba")[:, 1]
    final = LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000).fit(X[lab], y1[lab])
    if (~lab).any():
        proba[~lab] = final.predict_proba(X[~lab])[:, 1]
    pred = (proba >= 0.5).astype(int)

    names = np.array(vec.get_feature_names_out())
    order = np.argsort(final.coef_[0])
    info = {
        "phuong_phap": "Học bán giám sát (Self-training + Logistic Regression)",
        "nhan_yeu_duong": int((y0 == 1).sum()), "nhan_yeu_am": int((y0 == 0).sum()),
        "chua_nhan": int((y0 == -1).sum()), "tu_gan_them": pseudo, "so_vong": n_iter,
        "nguong_tin_cay": threshold,
        "lat_nhan_duong_thanh_am": int(((y0 == 1) & (pred == 0)).sum()),
        "lat_nhan_am_thanh_duong": int(((y0 == 0) & (pred == 1)).sum()),
        "tu_huong_tngt": names[order[::-1][:25]].tolist(),
        "tu_huong_khong": names[order[:25]].tolist(),
    }
    return pred, proba, y0, info
