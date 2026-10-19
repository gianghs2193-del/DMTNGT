# -*- coding: utf-8 -*-
"""
tngt_features.py - Trich xuat SO BO cac truong tu van ban bai bao TNGT
(dung cho app hien thi; ket qua la nhan tu dong theo luat, can kiem tra lai tren mau).

Dau vao: DataFrame bai viet tu crawler (cot: source, url, published, title, sapo, content).
Dau ra : moi bai them cac cot: la_tngt, ly_do_loai, tinh_cu, tinh_moi, ngay_xay_ra, gio,
         khung_gio, thu, dip_le_tet, so_chet, so_bi_thuong, muc_do, loai_duong,
         cac co phuong tien / nguyen nhan / kieu tai nan / thoi tiet (0/1).
Them ham gop_vu() de gop cac bai viet ve cung mot vu (nhieu bao dua tin).
"""
import datetime as dt
import re
import unicodedata

import numpy as np
import pandas as pd


def nfc(s):
    return unicodedata.normalize("NFC", s or "") if isinstance(s, str) else ""


# =====================================================================
# 1. Tinh / thanh (63 tinh cu -> 34 tinh moi tu 1/7/2025)
# =====================================================================
# (bien the -> ten chuan 63 tinh)
PROVINCE_VARIANTS = {
    "Hà Nội": ["Hà Nội", "Thủ đô"],
    "TP.HCM": ["TP.HCM", "TP HCM", "TPHCM", "TP. HCM", "TP Hồ Chí Minh", "TP. Hồ Chí Minh",
               "thành phố Hồ Chí Minh", "Thành phố Hồ Chí Minh", "Sài Gòn"],
    "Hải Phòng": ["Hải Phòng"], "Đà Nẵng": ["Đà Nẵng"], "Cần Thơ": ["Cần Thơ"],
    "An Giang": ["An Giang"], "Bà Rịa - Vũng Tàu": ["Bà Rịa - Vũng Tàu", "Bà Rịa – Vũng Tàu",
                                                     "Bà Rịa-Vũng Tàu", "Vũng Tàu", "Bà Rịa"],
    "Bắc Giang": ["Bắc Giang"], "Bắc Kạn": ["Bắc Kạn", "Bắc Cạn"], "Bạc Liêu": ["Bạc Liêu"],
    "Bắc Ninh": ["Bắc Ninh"], "Bến Tre": ["Bến Tre"], "Bình Định": ["Bình Định"],
    "Bình Dương": ["Bình Dương"], "Bình Phước": ["Bình Phước"], "Bình Thuận": ["Bình Thuận"],
    "Cà Mau": ["Cà Mau"], "Cao Bằng": ["Cao Bằng"],
    "Đắk Lắk": ["Đắk Lắk", "Đăk Lăk", "Đắc Lắc", "Đắk Lăk"],
    "Đắk Nông": ["Đắk Nông", "Đăk Nông"], "Điện Biên": ["Điện Biên"], "Đồng Nai": ["Đồng Nai"],
    "Đồng Tháp": ["Đồng Tháp"], "Gia Lai": ["Gia Lai"], "Hà Giang": ["Hà Giang"],
    "Hà Nam": ["Hà Nam"], "Hà Tĩnh": ["Hà Tĩnh"], "Hải Dương": ["Hải Dương"],
    "Hậu Giang": ["Hậu Giang"], "Hòa Bình": ["Hòa Bình", "Hoà Bình"], "Hưng Yên": ["Hưng Yên"],
    "Khánh Hòa": ["Khánh Hòa", "Khánh Hoà"], "Kiên Giang": ["Kiên Giang"], "Kon Tum": ["Kon Tum"],
    "Lai Châu": ["Lai Châu"], "Lâm Đồng": ["Lâm Đồng"], "Lạng Sơn": ["Lạng Sơn"],
    "Lào Cai": ["Lào Cai"], "Long An": ["Long An"], "Nam Định": ["Nam Định"],
    "Nghệ An": ["Nghệ An"], "Ninh Bình": ["Ninh Bình"], "Ninh Thuận": ["Ninh Thuận"],
    "Phú Thọ": ["Phú Thọ"], "Phú Yên": ["Phú Yên"], "Quảng Bình": ["Quảng Bình"],
    "Quảng Nam": ["Quảng Nam"], "Quảng Ngãi": ["Quảng Ngãi"], "Quảng Ninh": ["Quảng Ninh"],
    "Quảng Trị": ["Quảng Trị"], "Sóc Trăng": ["Sóc Trăng"], "Sơn La": ["Sơn La"],
    "Tây Ninh": ["Tây Ninh"], "Thái Bình": ["Thái Bình"], "Thái Nguyên": ["Thái Nguyên"],
    "Thanh Hóa": ["Thanh Hóa", "Thanh Hoá"],
    "Thừa Thiên Huế": ["Thừa Thiên Huế", "Thừa Thiên - Huế", "TP Huế", "Huế"],
    "Tiền Giang": ["Tiền Giang"], "Trà Vinh": ["Trà Vinh"], "Tuyên Quang": ["Tuyên Quang"],
    "Vĩnh Long": ["Vĩnh Long"], "Vĩnh Phúc": ["Vĩnh Phúc"], "Yên Bái": ["Yên Bái"],
}

# Sap xep don vi hanh chinh tu 1/7/2025: tinh cu -> tinh moi (34 tinh/thanh)
OLD_TO_NEW = {
    "Hà Giang": "Tuyên Quang", "Tuyên Quang": "Tuyên Quang",
    "Yên Bái": "Lào Cai", "Lào Cai": "Lào Cai",
    "Bắc Kạn": "Thái Nguyên", "Thái Nguyên": "Thái Nguyên",
    "Vĩnh Phúc": "Phú Thọ", "Hòa Bình": "Phú Thọ", "Phú Thọ": "Phú Thọ",
    "Bắc Giang": "Bắc Ninh", "Bắc Ninh": "Bắc Ninh",
    "Thái Bình": "Hưng Yên", "Hưng Yên": "Hưng Yên",
    "Hải Dương": "Hải Phòng", "Hải Phòng": "Hải Phòng",
    "Hà Nam": "Ninh Bình", "Nam Định": "Ninh Bình", "Ninh Bình": "Ninh Bình",
    "Quảng Bình": "Quảng Trị", "Quảng Trị": "Quảng Trị",
    "Quảng Nam": "Đà Nẵng", "Đà Nẵng": "Đà Nẵng",
    "Kon Tum": "Quảng Ngãi", "Quảng Ngãi": "Quảng Ngãi",
    "Bình Định": "Gia Lai", "Gia Lai": "Gia Lai",
    "Ninh Thuận": "Khánh Hòa", "Khánh Hòa": "Khánh Hòa",
    "Đắk Nông": "Lâm Đồng", "Bình Thuận": "Lâm Đồng", "Lâm Đồng": "Lâm Đồng",
    "Phú Yên": "Đắk Lắk", "Đắk Lắk": "Đắk Lắk",
    "Bình Dương": "TP.HCM", "Bà Rịa - Vũng Tàu": "TP.HCM", "TP.HCM": "TP.HCM",
    "Bình Phước": "Đồng Nai", "Đồng Nai": "Đồng Nai",
    "Long An": "Tây Ninh", "Tây Ninh": "Tây Ninh",
    "Sóc Trăng": "Cần Thơ", "Hậu Giang": "Cần Thơ", "Cần Thơ": "Cần Thơ",
    "Bến Tre": "Vĩnh Long", "Trà Vinh": "Vĩnh Long", "Vĩnh Long": "Vĩnh Long",
    "Tiền Giang": "Đồng Tháp", "Đồng Tháp": "Đồng Tháp",
    "Bạc Liêu": "Cà Mau", "Cà Mau": "Cà Mau",
    "Kiên Giang": "An Giang", "An Giang": "An Giang",
    "Hà Nội": "Hà Nội", "Thừa Thiên Huế": "Huế", "Lai Châu": "Lai Châu",
    "Điện Biên": "Điện Biên", "Sơn La": "Sơn La", "Lạng Sơn": "Lạng Sơn",
    "Quảng Ninh": "Quảng Ninh", "Thanh Hóa": "Thanh Hóa", "Nghệ An": "Nghệ An",
    "Hà Tĩnh": "Hà Tĩnh", "Cao Bằng": "Cao Bằng",
}

_VAR2CANON = {nfc(v): k for k, vs in PROVINCE_VARIANTS.items() for v in vs}
# so khop phan biet hoa/thuong (ten rieng viet hoa -> tranh "hòa bình" thuong), cum dai truoc
PROVINCE_RE = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(v) for v in sorted(_VAR2CANON, key=len, reverse=True)) + r")(?!\w)")
# tu dung truoc ten tinh nhung thuc ra la ten phuong/duong/cau... -> bo qua
_NOT_PROVINCE_PREV = {"phường", "xã", "đường", "phố", "cầu", "quận", "chợ", "bến", "khu", "ấp",
                      "thôn", "hẻm", "ngõ", "kiệt", "trấn", "hầm", "nút", "giao", "bùng", "binh"}


def find_province(text):
    for m in PROVINCE_RE.finditer(text):
        prev = text[max(0, m.start() - 12):m.start()].split()
        if prev and prev[-1].lower() in _NOT_PROVINCE_PREV:
            continue
        return _VAR2CANON[m.group(1)]
    return None


# =====================================================================
# 2. Tu dien dau hieu (co 0/1)
# =====================================================================
VEHICLES = {
    "xe máy": r"xe máy|xe mô ?tô|mô ?tô|môtô|xe gắn máy|xe tay ga|xe số",
    "xe đạp": r"xe đạp",
    "ô tô con": r"ô ?tô con|xe con|xe hơi|xe \d chỗ|xe bán tải|taxi|xe du lịch|ô ?tô(?! (?:tải|khách|đầu kéo|container|buýt|bus))",
    "xe tải": r"xe tải|ô ?tô tải|xe ben|xe bồn|xe ba gác|xe chở hàng",
    "container/đầu kéo": r"container|công-?te-?nơ|đầu kéo|rơ[ -]?moóc|sơ mi rơ moóc",
    "xe khách": r"xe khách|ô ?tô khách|xe giường nằm|xe buýt|xe bus|limousine|xe \d{2} chỗ|xe hợp đồng",
    "người đi bộ": r"người đi bộ|đi bộ qua đường|băng qua đường|sang đường",
}
CAUSES = {
    "rượu bia": r"nồng độ cồn|rượu bia|uống rượu|say xỉn|say rượu|có cồn",
    "ma túy": r"ma túy|ma tuý",
    "tốc độ cao": r"quá tốc độ|tốc độ cao|chạy nhanh|phóng nhanh",
    "vượt ẩu/lấn làn": r"vượt ẩu|vượt không đúng|vượt xe|lấn làn|lấn trái|lấn sang|lấn đường|chuyển làn",
    "mất lái": r"mất lái|mất kiểm soát|tự ngã|tự té",
    "mất phanh": r"mất phanh|mất thắng|hỏng phanh|bó phanh",
    "ngủ gật": r"ngủ gật|buồn ngủ",
    "đi ngược chiều": r"ngược chiều",
    "không đội MBH": r"không đội mũ bảo hiểm",
    "chưa có GPLX": r"không có giấy phép lái xe|chưa có giấy phép lái xe|không có bằng lái|chưa có bằng lái|chưa đủ tuổi",
    "dùng điện thoại": r"dùng điện thoại|sử dụng điện thoại|nghe điện thoại",
}
CRASH_TYPES = {
    "tông đuôi": r"(?:tông|đâm|húc) (?:vào )?(?:phía )?đuôi|tông từ phía sau|đâm từ phía sau",
    "va chạm trực diện": r"trực diện|đối đầu|đâm thẳng vào nhau|tông thẳng vào nhau",
    "lật/lao khỏi đường": r"lật|lao xuống|rơi xuống|lao khỏi|xuống vực|xuống mương|xuống sông",
    "liên hoàn": r"liên hoàn",
    "đâm vào vật cố định": r"(?:tông|đâm|lao|húc) (?:vào|thẳng vào) (?:cột|cây|nhà|quán|tường|dải phân cách|hộ lan|trụ|vách)",
}
WEATHER = {
    "trời mưa": r"trời mưa|mưa lớn|mưa to|đường trơn|trơn trượt|mưa rào",
    "sương mù": r"sương mù",
}
ROAD_TYPES = [
    ("cao tốc", r"cao tốc"),
    ("quốc lộ", r"quốc lộ|\bQL\s?\d"),
    ("tỉnh lộ", r"tỉnh lộ|đường tỉnh|\bĐT\s?\d"),
    ("đèo/đường núi", r"\bđèo\b"),
    ("đường nông thôn", r"liên xã|liên thôn|liên huyện|đường làng|đường thôn|đường huyện"),
    ("đường đô thị", r"\bphố\b|nội thành|nội đô|ngã tư|ngã ba|vòng xoay|nút giao"),
]

_compile = lambda d: {k: re.compile(v, re.I) for k, v in d.items()}
VEHICLE_RE, CAUSE_RE, CRASH_RE, WEATHER_RE = map(_compile, (VEHICLES, CAUSES, CRASH_TYPES, WEATHER))
ROAD_RE = [(k, re.compile(v, re.I)) for k, v in ROAD_TYPES]

# =====================================================================
# 3. Thuong vong
# =====================================================================
_NUMWORD = {"một": 1, "hai": 2, "ba": 3, "bốn": 4, "năm": 5, "sáu": 6, "bảy": 7, "tám": 8,
            "chín": 9, "mười": 10}
_N = (r"(?<![\d.,\w])(\d{1,3}|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười)(?![.,]\d)(?!\w)"
      # khong phai so nguoi: "13 tuoi", "5 km", "nam nay", "2 nam tu"...
      r"(?!\s*(?:tuổi|km|giờ|h\b|tấn|chỗ|mét|m\b|năm|tháng|ngày|phút|%|nay|qua|ngoái|trước|xe|vụ))")
# Dem so nguoi: so phai di kem DANH TU CHI NGUOI ("3 nguoi", "hai bo con", "4 thanh nien"...),
# tranh dem nham "lop 12", "29 Tet", "20 con heo", "QL54"...
PERSON = (r"(?:người|nạn nhân|em|cháu|bé|trẻ|học sinh|nam sinh|nữ sinh|sinh viên|thanh niên|thiếu niên|"
          r"công nhân|tài xế|lái xe|phụ xe|hành khách|du khách|khách|phụ nữ|đàn ông|cụ|ông|bà|chiến sĩ|"
          r"cán bộ|thành viên|học viên|bố con|cha con|mẹ con|vợ chồng|anh em|chị em|bà cháu|ông cháu|"
          r"mẹ|bố|cha|anh|chị|thợ|nhân viên|ngư dân|người dân)")
_FILL = r"\s+" + PERSON + r"(?:\s+[^\s\d,.;:!?()]+){0,4}?\s+(?:đã\s+|bị\s+|đều\s+)?"
DEATH_VERB = r"(?:tử vong|(?<!thoát )(?<!suýt )chết(?!\s*máy)|thiệt mạng|tử nạn|không qua khỏi)"
INJ_VERB = r"(?:bị thương|trọng thương|nhập viện|đi cấp cứu|chấn thương)"
DEATH_NUM = re.compile(_N + _FILL + DEATH_VERB, re.I)
INJ_NUM = re.compile(_N + _FILL + INJ_VERB, re.I)
MAX_CHET_MOT_VU, MAX_BI_THUONG_MOT_VU = 40, 100   # vuot qua -> gan nhu chac la so lieu tong hop
DEATH_ANY = re.compile(r"(?<!thoát )(?<!suýt )(?<!suýt bị )" + DEATH_VERB, re.I)
INJ_ANY = re.compile(INJ_VERB, re.I)


def _to_int(s):
    s = s.lower()
    return int(s) if s.isdigit() else _NUMWORD[s]


def count_casualties(head, body):
    """Uu tien tieu de + sapo; khong co thi tim trong ~1000 ky tu dau than bai."""
    res = {}
    for key, num_re, any_re in (("chet", DEATH_NUM, DEATH_ANY), ("bi_thuong", INJ_NUM, INJ_ANY)):
        val = None
        for zone in (head, body[:1000]):
            nums = [_to_int(m.group(1)) for m in num_re.finditer(zone)]
            if nums:
                val = max(nums)
                break
            if any_re.search(zone):
                val = 1  # co nhac den nhung khong neu so -> it nhat 1
                break
        res[key] = val
    return res


# =====================================================================
# 4. Thoi gian xay ra
# =====================================================================
PERIOD = r"(rạng sáng|sáng sớm|sáng|trưa|chiều|tối|đêm|khuya)"
DATE_RE = re.compile(PERIOD + r"?\s*(?:ngày\s+|hôm\s+)?(\d{1,2})[/-](\d{1,2})(?:[/-](\d{4}))?", re.I)
REL_RE = re.compile(r"(?:rạng sáng|sáng|trưa|chiều|tối|đêm|khuya|hôm)\s+(nay|qua)", re.I)
HOUR_RE = re.compile(r"(?:khoảng|lúc|vào|tầm|gần|hơn)\s+(\d{1,2})\s*(?:h|giờ)\s*(\d{1,2})?(?:\s*phút)?"
                     r"(?:\s+(sáng|trưa|chiều|tối|đêm|khuya))?", re.I)
HOUR_BARE_RE = re.compile(r"(?<![\d/.,\w])(\d{1,2})h(\d{2})?(?![\w/])", re.I)
PERIOD_ONLY_RE = re.compile(r"(rạng sáng|sáng sớm|sáng|trưa|chiều|tối|đêm|khuya)\s+(?:nay|qua|\d{1,2}/\d{1,2}|ngày)", re.I)

TIME_BUCKETS = ["Sáng (5–10h)", "Trưa (11–13h)", "Chiều (14–17h)", "Tối (18–21h)", "Đêm khuya (22–4h)"]
WEEKDAYS = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ nhật"]
_PERIOD_BUCKET = {"rạng sáng": 4, "sáng sớm": 0, "sáng": 0, "trưa": 1, "chiều": 2, "tối": 3,
                  "đêm": 4, "khuya": 4}


def _hour_bucket(h):
    if 5 <= h <= 10:
        return 0
    if 11 <= h <= 13:
        return 1
    if 14 <= h <= 17:
        return 2
    if 18 <= h <= 21:
        return 3
    return 4


def find_hour(text):
    m = HOUR_RE.search(text)
    h, period = None, None
    if m:
        h, period = int(m.group(1)), (m.group(3) or "").lower()
    else:
        m = HOUR_BARE_RE.search(text)
        if m:
            h = int(m.group(1))
    if h is None or h > 23:
        return None
    if period in ("chiều", "tối") and h < 12:
        h += 12
    elif period in ("đêm", "khuya") and 6 <= h < 12:
        h += 12
    elif period == "trưa" and h < 5:
        h += 12
    return h


def find_event_date(text, pub):
    """Ngay xay ra: 'toi 15/10' -> 15/10 (nam suy tu ngay dang); 'dem qua' -> pub-1; mac dinh = pub."""
    if pub is None or pd.isna(pub):
        return None, "khong_ro"
    pub_d = pub.date()
    for m in DATE_RE.finditer(text):
        d, mo, y = int(m.group(2)), int(m.group(3)), m.group(4)
        try:
            cand = dt.date(int(y) if y else pub_d.year, mo, d)
        except ValueError:
            continue
        if not y and cand > pub_d + dt.timedelta(days=2):
            try:
                cand = dt.date(pub_d.year - 1, mo, d)
            except ValueError:
                continue
        if pub_d - dt.timedelta(days=45) <= cand <= pub_d + dt.timedelta(days=1):
            return cand, "trong_bai"
    m = REL_RE.search(text)
    if m:
        return (pub_d - dt.timedelta(days=1) if m.group(1).lower() == "qua" else pub_d), "tuong_doi"
    return pub_d, "ngay_dang"


def _holiday_dates():
    tet = [dt.date(2023, 1, 22), dt.date(2024, 2, 10), dt.date(2025, 1, 29), dt.date(2026, 2, 17),
           dt.date(2027, 2, 6)]
    days = set()
    for t in tet:
        days.update(t + dt.timedelta(days=i) for i in range(-7, 8))
    for y in range(2023, 2028):
        for base in (dt.date(y, 1, 1), dt.date(y, 4, 30), dt.date(y, 9, 2)):
            days.update(base + dt.timedelta(days=i) for i in range(-2, 3))
    return days


HOLIDAYS = _holiday_dates()

# =====================================================================
# 5. Loc bai la tuong thuat mot vu TNGT duong bo
# =====================================================================
ACCIDENT_RE = re.compile(
    r"tai nạn|va chạm|(?:tông|đâm|húc|cán|tuột dốc)\s|lật xe|xe lật|bị lật|lao (?:vào|xuống|qua|khỏi)|"
    r"liên hoàn|mất lái|mất phanh|cuốn vào gầm|tử vong|thiệt mạng|tử nạn", re.I)
EXCLUDE_TITLE_RE = re.compile(
    r"khởi tố|truy tố|xét xử|tuyên án|hầu tòa|phạt tù|bắt tạm giam|xử phạt|phạt nguội|bị phạt|"
    r"thống kê|toàn quốc|cả nước|\d+ tháng|kéo giảm|giảm \d|kỷ niệm|tưởng niệm|"
    r"tàu hỏa|tàu hoả|đường sắt|đường ngang|máy bay|trực thăng|tàu cá|ca nô|sà lan|chìm|đuối nước|"
    r"điện giật|tai nạn lao động|cháy nhà|hỏa hoạn|metro|clip|video|vì sao|bao nhiêu|có được|"
    r"hướng dẫn|quy định|đề xuất|thông tư|nghị định|bảo hiểm|bồi thường",
    re.I)
FOREIGN_RE = re.compile(
    r"(?<!\w)(Mỹ|Trung Quốc|Ấn Độ|Thái Lan|Hàn Quốc|Nhật Bản|Indonesia|Malaysia|Philippines|Nga|Pháp|"
    r"Đức|Anh|Brazil|Peru|Ai Cập|Pakistan|Campuchia|Lào|Myanmar|Australia|Úc|Canada|Mexico)"
    r"(?!\s+[A-ZĐÀ-Ỹ])(?!\w)")


def classify(head, prov, deaths, injuries):
    if not ACCIDENT_RE.search(head):
        return 0, "khong co dau hieu tai nan"
    m = EXCLUDE_TITLE_RE.search(head[:200])
    if m:
        return 0, f"tu khoa loai tru: {m.group(0)}"
    if prov is None and FOREIGN_RE.search(head):
        return 0, "tin nuoc ngoai"
    if (deaths or 0) > 60 or (injuries or 0) > 150:
        return 0, "so lieu tong hop"
    if deaths is None and injuries is None and not re.search(
            r"va chạm|tông|đâm|lật|lao|liên hoàn|cán|húc", head, re.I):
        return 0, "khong ro hau qua"
    return 1, ""


# =====================================================================
# 6. Ham chinh
# =====================================================================
def extract_features(df):
    """Trich xuat truong + nhan cua bo loc luat (la_tngt_luat). Loc bang hoc may: xem tngt_ml.py."""
    df = df.copy()
    for c in ("title", "sapo", "content"):
        df[c] = df[c].map(nfc) if c in df else ""
    df["published"] = pd.to_datetime(df["published"], errors="coerce", utc=True).dt.tz_convert(
        "Asia/Ho_Chi_Minh").dt.tz_localize(None)

    rows = []
    for title, sapo, content, pub in zip(df["title"], df["sapo"], df["content"], df["published"]):
        head = f"{title}. {sapo}"
        near = f"{head} {content[:1500]}"
        full = f"{head} {content}"
        prov = find_province(head) or find_province(content[:1500])
        cas = count_casualties(head, content)
        la, why = classify(head, prov, cas["chet"], cas["bi_thuong"])
        date, date_src = find_event_date(f"{head} {content[:600]}", pub)
        hour = find_hour(f"{head} {content[:600]}")
        if hour is not None:
            bucket = TIME_BUCKETS[_hour_bucket(hour)]
        else:
            m = PERIOD_ONLY_RE.search(f"{head} {content[:600]}")
            bucket = TIME_BUCKETS[_PERIOD_BUCKET[m.group(1).lower()]] if m else None
        road = next((k for k, r in ROAD_RE if r.search(near)), "khác/không rõ")
        rec = {
            "la_tngt": la, "ly_do_loai": why,
            "tinh_cu": prov, "tinh_moi": OLD_TO_NEW.get(prov) if prov else None,
            "ngay_xay_ra": pd.Timestamp(date) if date else pd.NaT, "nguon_ngay": date_src,
            "gio": hour, "khung_gio": bucket,
            "so_chet": cas["chet"], "so_bi_thuong": cas["bi_thuong"],
            "loai_duong": road,
        }
        for k, r in VEHICLE_RE.items():
            rec[f"pt_{k}"] = int(bool(r.search(near)))
        for k, r in CAUSE_RE.items():
            rec[f"nn_{k}"] = int(bool(r.search(full)))
        for k, r in CRASH_RE.items():
            rec[f"kieu_{k}"] = int(bool(r.search(near)))
        for k, r in WEATHER_RE.items():
            rec[f"tt_{k}"] = int(bool(r.search(full)))
        rows.append(rec)

    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    out["la_tngt_luat"] = out["la_tngt"]
    out["ly_do_loai_luat"] = out["ly_do_loai"]
    out["co_tu_khoa_tn"] = [bool(ACCIDENT_RE.search(f"{t}. {s}")) for t, s in zip(out["title"], out["sapo"])]
    out["p_tngt"] = np.nan
    out["phuong_phap_loc"] = "luật"
    d = out["ngay_xay_ra"]
    out["thu"] = pd.Categorical(d.dt.dayofweek.map(lambda i: WEEKDAYS[int(i)] if pd.notna(i) else None),
                                categories=WEEKDAYS, ordered=True)
    out["thang"] = d.dt.to_period("M").dt.to_timestamp()
    out["dip_le_tet"] = d.dt.date.map(lambda x: int(x in HOLIDAYS) if pd.notna(x) else 0)
    out["khung_gio"] = pd.Categorical(out["khung_gio"], categories=TIME_BUCKETS, ordered=True)
    out["muc_do"] = np.select(
        [out["so_chet"].fillna(0) > 0, out["so_bi_thuong"].fillna(0) > 0],
        ["Có người chết", "Chỉ bị thương"], default="Không rõ/không thương vong")
    out["co_tu_vong"] = (out["so_chet"].fillna(0) > 0).astype(int)
    return out


# =====================================================================
# 7. Gop cac bai cung mot vu
# =====================================================================
def _shingles(text):
    w = re.findall(r"\w+", text.lower())
    return {" ".join(w[i:i + 2]) for i in range(len(w) - 1)}


# ---------------------------------------------------------------------
# Ham bam (hashing) de kiem tra trung lap
# ---------------------------------------------------------------------
def normalize_for_hash(text):
    """Chuan hoa truoc khi bam: NFC, chu thuong, bo dau cau, gop khoang trang."""
    return " ".join(re.findall(r"\w+", nfc(text).lower()))


def content_hash(title, content):
    """SHA-1 cua noi dung da chuan hoa -> 2 bai giong het nhau (khac URL) co cung ma bam."""
    import hashlib
    return hashlib.sha1(normalize_for_hash(f"{title} {content}").encode("utf-8")).hexdigest()


def loai_trung_hash(df):
    """Muc 1 - trung HOAN TOAN: bang bam (dict ma_bam -> bai dau tien), O(n).
    Tra ve (df_da_loai, so_bai_bi_loai)."""
    df = df.copy()
    df["ma_bam"] = [content_hash(t, c) for t, c in zip(df["title"], df["content"])]
    seen = {}            # bang bam: ma_bam -> url dau tien
    keep = []
    for h, u in zip(df["ma_bam"], df["url"]):
        keep.append(h not in seen)
        seen.setdefault(h, u)
    out = df[keep]
    return out, int(len(df) - len(out))


_P = np.uint64((1 << 31) - 1)          # so nguyen to Mersenne 2^31-1
_rng = np.random.default_rng(12345)
N_PERM, BANDS = 128, 64                # 128 ham bam, chia 64 bang x 2 hang
ROWS = N_PERM // BANDS
_A = _rng.integers(1, int(_P), N_PERM, dtype=np.uint64)
_B = _rng.integers(0, int(_P), N_PERM, dtype=np.uint64)


def minhash_signature(shingles):
    """Chu ky MinHash: voi moi ham bam h_i(x) = (a_i*x + b_i) mod p, lay gia tri NHO NHAT tren
    tap shingle. P(chu ky trung tai 1 vi tri) = do tuong dong Jaccard cua 2 tap."""
    import zlib
    if not shingles:
        return np.full(N_PERM, _P, dtype=np.uint64)
    x = np.array([zlib.crc32(s.encode("utf-8")) & 0x7FFFFFFF for s in shingles], dtype=np.uint64)
    return ((_A[:, None] * x[None, :] + _B[:, None]) % _P).min(axis=1)


def lsh_candidates(signatures):
    """LSH (Locality-Sensitive Hashing): chia chu ky thanh BANDS bang, bam moi bang vao thung;
    2 bai roi chung thung o it nhat 1 bang -> cap ung vien. Nguong ~ (1/BANDS)^(1/ROWS)."""
    from collections import defaultdict
    pairs = set()
    for b in range(BANDS):
        buckets = defaultdict(list)                  # bang bam: khoa bang -> danh sach bai
        for i, sig in enumerate(signatures):
            buckets[sig[b * ROWS:(b + 1) * ROWS].tobytes()].append(i)
        for members in buckets.values():
            if 1 < len(members) <= 200:
                for x in range(len(members)):
                    for y in range(x + 1, len(members)):
                        pairs.add((members[x], members[y]))
    return pairs


def gop_vu(acc, sim_threshold=0.22):
    """acc: cac bai la_tngt=1. Gop cac bai cung mot vu:
      1) MinHash + LSH sinh cap ung vien (thay vi so sanh moi cap),
      2) kiem tra cap ung vien: cach nhau <=1 ngay, cung tinh, Jaccard >= nguong,
      3) Union-Find gom thanh vu.
    Tra ve (bang_vu, acc_co_vu_id); thong ke dat trong bang_vu.attrs["kiem_tra_trung"]."""
    acc = acc.reset_index(drop=True).copy()
    n = len(acc)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    sh = [_shingles(f"{t} {s}") for t, s in zip(acc["title"], acc["sapo"])]
    sigs = [minhash_signature(x) for x in sh]
    cand_lsh = lsh_candidates(sigs)
    dates = acc["ngay_xay_ra"].values.astype("datetime64[D]")
    # Blocking bang bam: khoa (tinh, ngay) -> bai; ghep voi ngay ke tiep de bat bai cung tinh,
    # lech 1 ngay, cung so nguoi chet nhung viet khac nhau (Jaccard thap, LSH co the bo sot)
    from collections import defaultdict
    blocks = defaultdict(list)
    for i, (pv, d) in enumerate(zip(acc["tinh_moi"], dates)):
        if pv and not np.isnat(d):
            blocks[(pv, d)].append(i)
    cand_block = set()
    for (pv, d), mem in blocks.items():
        nxt = blocks.get((pv, d + np.timedelta64(1, "D")), [])
        group = mem + nxt
        for x in range(len(mem)):
            for y in range(x + 1, len(group)):
                cand_block.add(tuple(sorted((mem[x], group[y]))))
    # Khoi bam thu 2: khoa (tinh, so nguoi chet) voi vu tu 3 nguoi chet tro len -> gom cac bai
    # dua tin tiep (nhan chung, nguyen nhan...) dang cach vai ngay, tieu de khac han nhau
    sev = defaultdict(list)
    for i, (pv, d, k) in enumerate(zip(acc["tinh_moi"], dates, acc["so_chet"])):
        if pv and not np.isnat(d) and pd.notna(k) and k >= 3:
            sev[(pv, int(k))].append(i)
    cand_sev = set()
    for mem in sev.values():
        for x in range(len(mem)):
            for y in range(x + 1, len(mem)):
                if abs((dates[mem[x]] - dates[mem[y]]).astype(int)) <= 7:
                    cand_sev.add(tuple(sorted((mem[x], mem[y]))))
    cands = cand_lsh | cand_block | cand_sev
    merged_pairs = 0
    for i, j in cands:
        if (i, j) in cand_sev or (j, i) in cand_sev:     # cung tinh, cung so chet >= 3, <= 7 ngay
            parent[find(i)] = find(j)
            merged_pairs += 1
            continue
        if np.isnat(dates[i]) or np.isnat(dates[j]) or abs((dates[j] - dates[i]).astype(int)) > 1:
            continue
        pi, pj = acc.at[i, "tinh_moi"], acc.at[j, "tinh_moi"]
        if pi and pj and pi != pj:
            continue
        sim = len(sh[i] & sh[j]) / (len(sh[i] | sh[j]) or 1)
        di, dj = acc.at[i, "so_chet"], acc.at[j, "so_chet"]
        same_deaths = pd.notna(di) and pd.notna(dj) and di == dj and di > 0
        if sim >= sim_threshold or (same_deaths and pi and pi == pj and sim >= 0.1):
            parent[find(i)] = find(j)
            merged_pairs += 1
    stats = {"so_bai": n, "cap_ung_vien_lsh": len(cand_lsh), "cap_ung_vien_khoi_tinh_ngay": len(cand_block),
             "cap_ung_vien_khoi_tinh_so_chet": len(cand_sev),
             "cap_ung_vien_tong": len(cands), "cap_so_sanh_neu_vet_can": n * (n - 1) // 2,
             "cap_duoc_gop": merged_pairs, "so_ham_bam": N_PERM, "so_bang": BANDS, "so_hang_moi_bang": ROWS,
             "nguong_lsh_xap_xi": round((1 / BANDS) ** (1 / ROWS), 3)}

    acc["vu_id"] = [find(i) for i in range(n)]
    acc["_len"] = acc["content"].str.len()
    rep = acc.sort_values("_len", ascending=False).drop_duplicates("vu_id").set_index("vu_id")
    agg = acc.groupby("vu_id").agg(
        so_bai=("url", "size"),
        cac_nguon=("source", lambda s: ", ".join(sorted(set(s)))),
        so_chet_max=("so_chet", "max"),
        so_bi_thuong_max=("so_bi_thuong", "max"),
    )
    flag_cols = [c for c in acc.columns if c.startswith(("pt_", "nn_", "kieu_", "tt_"))]
    flags = acc.groupby("vu_id")[flag_cols].max()
    vu = rep.drop(columns=flag_cols + ["_len"]).join(agg).join(flags)
    vu["so_chet"] = vu["so_chet_max"]
    vu["so_bi_thuong"] = vu["so_bi_thuong_max"]
    vu = vu.drop(columns=["so_chet_max", "so_bi_thuong_max"])
    vu["co_tu_vong"] = (vu["so_chet"].fillna(0) > 0).astype(int)
    vu["muc_do"] = np.select([vu["so_chet"].fillna(0) > 0, vu["so_bi_thuong"].fillna(0) > 0],
                             ["Có người chết", "Chỉ bị thương"], default="Không rõ/không thương vong")
    vu = vu.reset_index()
    stats["so_vu"] = len(vu)
    vu.attrs["kiem_tra_trung"] = stats
    return vu, acc.drop(columns=["_len"])


# =====================================================================
# 8. Luat ket hop (Apriori don gian, toi da 3 hang muc ve trai)
# =====================================================================
def build_items(vu):
    items = {}
    for c in vu.columns:
        if c.startswith(("pt_", "nn_", "kieu_", "tt_")):
            items[c.split("_", 1)[1]] = vu[c].astype(bool)
    for b in TIME_BUCKETS:
        items[f"giờ: {b}"] = vu["khung_gio"].astype(object) == b
    for r in vu["loai_duong"].dropna().unique():
        if r not in ("khác/không rõ", "Không rõ"):
            items[f"đường: {r}"] = vu["loai_duong"] == r
    items["cuối tuần"] = vu["thu"].isin(["Thứ 7", "Chủ nhật"])
    items["dịp lễ/Tết"] = vu["dip_le_tet"].astype(bool)
    items["CÓ TỬ VONG"] = vu["co_tu_vong"].astype(bool)
    return pd.DataFrame(items).fillna(False).astype(bool)


def apriori_rules(items, min_support=0.03, min_conf=0.5, max_len=3, target=None):
    n = len(items)
    if n == 0:
        return pd.DataFrame()
    supp1 = items.mean()
    freq = {frozenset([c]): s for c, s in supp1.items() if s >= min_support}
    level = list(freq)
    cols = {c: items[c].values for c in items.columns}
    for k in range(2, max_len + 1):
        cands = set()
        for i, a in enumerate(level):
            for b in level[i + 1:]:
                u = a | b
                if len(u) == k:
                    cands.add(u)
        level = []
        for c in cands:
            mask = np.logical_and.reduce([cols[x] for x in c])
            s = mask.mean()
            if s >= min_support:
                freq[c] = s
                level.append(c)
        if not level:
            break
    rules = []
    for itemset, s in freq.items():
        if len(itemset) < 2:
            continue
        conseqs = [target] if target else list(itemset)
        for cons in conseqs:
            if cons not in itemset:
                continue
            ante = itemset - {cons}
            sa = freq.get(ante)
            if not sa:
                continue
            conf = s / sa
            if conf >= min_conf:
                rules.append({"Nếu (vế trái)": " + ".join(sorted(ante)), "Thì": cons,
                              "support": s, "confidence": conf, "lift": conf / supp1[cons],
                              "số vụ": int(round(s * n))})
    return pd.DataFrame(rules).sort_values(["lift", "confidence"], ascending=False) if rules else pd.DataFrame()
