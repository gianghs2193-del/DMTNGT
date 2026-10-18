#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crawl_tngt.py - Cao du lieu THO tin tai nan giao thong (TNGT) duong bo tu 4 bao lon
theo phuong phap duyet PATH: trang chuyen muc -> phan trang -> tung bai viet.

De tai: Phan tich cac yeu to gay thuong vong trong TNGT duong bo toan quoc.

Giai doan nay CHI thu thap du lieu tho: HTML goc (nen .gz) + van ban chinh + metadata.
Loc bai dung la TNGT va trich xuat truong (thoi gian, dia diem, phuong tien,
thuong vong, nguyen nhan, thoi tiet...) lam o buoc sau, tren du lieu tho nay.
Cot `is_candidate` chi la co danh dau so bo theo tu khoa, KHONG dung de loai bai.

Nguon (da xac minh phan trang theo path, thang 10/2026):
    VnExpress   https://vnexpress.net/thoi-su/giao-thong        (-p2, -p3, ...)
    Dan Tri     https://dantri.com.vn/thoi-su/giao-thong.htm    (/trang-2.htm, ...)
    VietNamNet  https://vietnamnet.vn/thoi-su/giao-thong        (-page1, -page2, ...)
    Znews       https://znews.vn/giao-thong.html                (-trang2.html, ...)

Cai dat:
    pip install requests beautifulsoup4 trafilatura

Cach chay:
    python crawl_tngt.py run --sources vnexpress --max-pages 3   # chay thu nho truoc
    python crawl_tngt.py run                       # cao ca 4 nguon
    python crawl_tngt.py run --restart             # duyet lai tu trang 1
    python crawl_tngt.py fetch                     # tai lai bai con pending/loi
    python crawl_tngt.py stats                     # thong ke
    python crawl_tngt.py export                    # xuat data/raw_articles.csv

CHO CHAN (de khong treo/chet may):
    - Ctrl+C mot lan: dung an toan sau bai hien tai (Ctrl+C lan 2: thoat ngay).
    - Tao file data/STOP (vd: `touch data/STOP`): dung an toan, ke ca khi chay nen.
    - Gioi han so trang / so bai / thoi gian chay / dung luong o dia (CONFIG).
    - Moi request co timeout va gioi han kich thuoc tai ve.
    - Dung nguon khi: N trang lien tiep khong co link moi, trang bi lap lai
      (het phan trang), gap N bai lien tiep cu hon moc thoi gian, loi lien tiep,
      hoac bi chan (429/403) qua nhieu lan.
    - Ghi tung bai xuong SQLite ngay, chay lai se tiep tuc tu cho dung.
"""

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import logging
import os
import random
import re
import shutil
import signal
import sqlite3
import sys
import time
import unicodedata
from urllib import robotparser
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import trafilatura
except ImportError:  # van chay duoc, chi mat phan boc noi dung chinh
    trafilatura = None


# =====================================================================
# CAU HINH - chinh o day
# =====================================================================
CONFIG = {
    "data_dir": "data",
    # Bai dang truoc moc nay coi la ngoai pham vi (dung de dung nguon).
    "stop_before": "2023-01-01",
    # --- Cho chan toan cuc ---
    "max_pages_per_source": 400,      # so trang danh sach toi da moi nguon
    "max_articles_per_run": 6000,     # so bai tai toi da moi lan chay
    "max_runtime_minutes": 240,       # chay qua thoi gian nay thi dung
    "min_free_disk_gb": 1.0,          # o dia con it hon muc nay thi dung
    # --- Cho chan theo nguon ---
    "old_streak_to_stop": 25,         # N bai lien tiep cu hon moc -> dung nguon
    "empty_pages_to_stop": 3,         # N trang lien tiep khong co link moi -> dung nguon
    "max_consecutive_errors": 10,     # N loi lien tiep -> dung nguon
    "max_blocks_per_source": 3,       # bi 429/403 qua N lan -> dung nguon
    "block_cooldown_seconds": 90,     # nghi bao lau khi bi 429/403 (nhan theo so lan)
    "max_links_per_page": 200,        # chan trang bat thuong co qua nhieu link
    # --- Mang ---
    "delay_range": (1.2, 2.8),        # nghi ngau nhien giua 2 request (giay)
    "timeout": (10, 25),              # (connect, read) giay
    "max_bytes": 5 * 1024 * 1024,     # khong tai trang lon hon 5 MB
    "max_attempts_per_url": 3,
    "user_agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 "
        "(student data-mining project; polite crawler)"
    ),
}


def _paged(base, fmt):
    """Tao ham page_url(n): trang 1 la base, trang n>=2 theo fmt."""
    return lambda n: base if n == 1 else fmt.format(n=n, m=n - 1)


SOURCES = {
    "vnexpress": {
        "name": "VnExpress",
        "page_url": _paged("https://vnexpress.net/thoi-su/giao-thong",
                           "https://vnexpress.net/thoi-su/giao-thong-p{n}"),
        "article_re": r"^https://vnexpress\.net/[a-z0-9-]+-\d{7}\.html$",
        "exclude_re": None,
    },
    "dantri": {
        "name": "Dan Tri",
        "page_url": _paged("https://dantri.com.vn/thoi-su/giao-thong.htm",
                           "https://dantri.com.vn/thoi-su/giao-thong/trang-{n}.htm"),
        "article_re": r"^https://dantri\.com\.vn/[a-z0-9-]+/[a-z0-9-]+-\d{17}\.htm$",
        "exclude_re": r"dantri\.com\.vn/(?:video|photo-news|infographic)/",
    },
    "vietnamnet": {
        "name": "VietNamNet",
        # trang 1 = base; trang n>=2 = -page{n-1} (site danh so tu page0)
        "page_url": _paged("https://vietnamnet.vn/thoi-su/giao-thong",
                           "https://vietnamnet.vn/thoi-su/giao-thong-page{m}"),
        "article_re": r"^https://vietnamnet\.vn/[a-z0-9-]+-\d{6,8}\.html$",
        "exclude_re": None,
    },
    "znews": {
        "name": "Znews",
        "page_url": _paged("https://znews.vn/giao-thong.html",
                           "https://znews.vn/giao-thong-trang{n}.html"),
        "article_re": r"^https://(?:lifestyle\.zingnews\.vn|znews\.vn)/[a-z0-9-]+-post\d+\.html$",
        "exclude_re": r"/video-",
    },
}

# Tu khoa danh dau so bo (KHONG dung de loai bai, chi ghi vao cot is_candidate).
CANDIDATE_PATTERNS = [
    r"tai nạn", r"va chạm", r"tông (?:vào|chết|trúng|xe|người)", r"đâm (?:vào|xe|trúng)",
    r"lật xe", r"lao xuống", r"mất lái", r"mất phanh", r"liên hoàn", r"tử vong",
    r"thiệt mạng", r"tử nạn", r"bị thương", r"thương vong", r"cán (?:chết|qua|người)",
    r"cuốn vào gầm",
]


# =====================================================================
# Tien ich
# =====================================================================
log = logging.getLogger("tngt")


def setup_logging(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    fh = logging.FileHandler(os.path.join(data_dir, "crawl.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(ch)
    log.addHandler(fh)


def now_iso():
    return dt.datetime.now().isoformat(timespec="seconds")


def norm_text(s):
    return unicodedata.normalize("NFC", s or "").strip()


def clean_url(url):
    """Bo query va fragment (#box_comment, ?utm=...) de khu trung lap URL."""
    p = urlsplit(url)
    return urlunsplit((p.scheme, p.netloc.lower(), p.path, "", ""))


def parse_date(value):
    """Doi chuoi ngay (ISO, YYYY-MM-DD, dd/mm/yyyy) thanh datetime; None neu khong doc duoc."""
    if not value:
        return None
    s = str(value).strip()
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        pass
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        try:
            return dt.datetime(int(m[1]), int(m[2]), int(m[3]))
        except ValueError:
            return None
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", s)
    if m:
        try:
            return dt.datetime(int(m[3]), int(m[2]), int(m[1]))
        except ValueError:
            return None
    return None


# =====================================================================
# Co dung (Ctrl+C, file STOP, thoi gian, so bai, o dia)
# =====================================================================
class StopController:
    def __init__(self, cfg):
        self.cfg = cfg
        self.flag = False
        self.reason = ""
        self.sigint_hits = 0
        self.started = time.time()
        self.articles_done = 0
        self.stop_file = os.path.join(cfg["data_dir"], "STOP")
        if os.path.exists(self.stop_file):
            os.remove(self.stop_file)  # don file STOP cu cua lan truoc

    def install_signal(self):
        def handler(signum, frame):
            self.sigint_hits += 1
            if self.sigint_hits >= 2:
                log.warning("Ctrl+C lan 2: thoat ngay.")
                raise KeyboardInterrupt
            self.request("nguoi dung bam Ctrl+C")
            log.warning("Se dung sau bai hien tai. Bam Ctrl+C lan nua de thoat ngay.")
        signal.signal(signal.SIGINT, handler)

    def request(self, reason):
        if not self.flag:
            self.flag = True
            self.reason = reason

    def should_stop(self):
        if self.flag:
            return True
        if os.path.exists(self.stop_file):
            self.request("phat hien file data/STOP")
        elif time.time() - self.started > self.cfg["max_runtime_minutes"] * 60:
            self.request(f"qua {self.cfg['max_runtime_minutes']} phut")
        elif self.articles_done >= self.cfg["max_articles_per_run"]:
            self.request(f"da tai du {self.cfg['max_articles_per_run']} bai")
        else:
            free_gb = shutil.disk_usage(self.cfg["data_dir"]).free / 1024 ** 3
            if free_gb < self.cfg["min_free_disk_gb"]:
                self.request(f"o dia chi con {free_gb:.2f} GB")
        return self.flag

    def sleep(self, seconds):
        """Ngu tung doan ngan de Ctrl+C / file STOP co hieu luc ngay."""
        end = time.time() + seconds
        while time.time() < end:
            if self.should_stop():
                return
            remaining = end - time.time()
            if remaining <= 0:  # should_stop() ton thoi gian, co the da qua han
                return
            time.sleep(min(0.5, remaining))


# =====================================================================
# Tai trang (timeout, gioi han kich thuoc, robots.txt, nghi giua request)
# =====================================================================
class Fetcher:
    def __init__(self, cfg, stopper):
        self.cfg = cfg
        self.stopper = stopper
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": cfg["user_agent"],
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.5",
        })
        retry = Retry(total=2, backoff_factor=2, status_forcelist=[500, 502, 503, 504],
                      allowed_methods=["GET"], raise_on_status=False)
        adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.robots = {}
        self.last_request = 0.0

    def _wait(self):
        lo, hi = self.cfg["delay_range"]
        gap = random.uniform(lo, hi) - (time.time() - self.last_request)
        if gap > 0:
            self.stopper.sleep(gap)

    def allowed(self, url):
        p = urlsplit(url)
        host = f"{p.scheme}://{p.netloc}"
        if host not in self.robots:
            rp = robotparser.RobotFileParser()
            try:
                r = self.session.get(host + "/robots.txt", timeout=self.cfg["timeout"])
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])
            except requests.RequestException as e:
                log.warning("Khong doc duoc robots.txt cua %s (%s), tam coi la cho phep", host, e)
                rp.parse([])
            self.robots[host] = rp
        return self.robots[host].can_fetch(self.cfg["user_agent"], url)

    def get(self, url):
        """Tra ve dict: status ('ok'|'http'|'blocked'|'robots'|'too_large'|'not_html'|'net'),
        code, html, final_url, error."""
        if not self.allowed(url):
            return {"status": "robots", "code": None, "html": None, "final_url": url,
                    "error": "robots.txt khong cho phep"}
        self._wait()
        self.last_request = time.time()
        try:
            with self.session.get(url, timeout=self.cfg["timeout"], stream=True,
                                  allow_redirects=True) as r:
                code = r.status_code
                if code in (403, 429):
                    return {"status": "blocked", "code": code, "html": None,
                            "final_url": r.url, "error": f"HTTP {code}"}
                if code != 200:
                    return {"status": "http", "code": code, "html": None,
                            "final_url": r.url, "error": f"HTTP {code}"}
                ctype = r.headers.get("Content-Type", "")
                if "html" not in ctype.lower():
                    return {"status": "not_html", "code": code, "html": None,
                            "final_url": r.url, "error": f"Content-Type {ctype}"}
                chunks, size = [], 0
                for chunk in r.iter_content(64 * 1024):
                    size += len(chunk)
                    if size > self.cfg["max_bytes"]:
                        return {"status": "too_large", "code": code, "html": None,
                                "final_url": r.url, "error": f"> {self.cfg['max_bytes']} bytes"}
                    chunks.append(chunk)
                enc = r.encoding if r.encoding and r.encoding.lower() != "iso-8859-1" else "utf-8"
                html = b"".join(chunks).decode(enc, errors="replace")
                return {"status": "ok", "code": code, "html": html,
                        "final_url": r.url, "error": None}
        except requests.RequestException as e:
            return {"status": "net", "code": None, "html": None, "final_url": url,
                    "error": f"{type(e).__name__}: {e}"[:300]}


# =====================================================================
# Luu tru SQLite
# =====================================================================
SCHEMA = """
CREATE TABLE IF NOT EXISTS progress (
    source TEXT PRIMARY KEY,
    last_page INTEGER DEFAULT 0,
    finished INTEGER DEFAULT 0,
    stop_reason TEXT,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS urls (
    url TEXT PRIMARY KEY,
    source TEXT,
    title_hint TEXT,
    list_page INTEGER,
    status TEXT DEFAULT 'pending',   -- pending | done | old | error | failed | skipped
    attempts INTEGER DEFAULT 0,
    http_code INTEGER,
    error TEXT,
    discovered_at TEXT,
    fetched_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_urls_status ON urls(status, source);
CREATE TABLE IF NOT EXISTS articles (
    url TEXT PRIMARY KEY,
    source TEXT,
    final_url TEXT,
    title TEXT,
    sapo TEXT,
    content TEXT,
    published TEXT,
    category TEXT,
    author TEXT,
    word_count INTEGER,
    is_candidate INTEGER,
    candidate_hits TEXT,
    html_path TEXT,
    fetched_at TEXT
);
"""


class Store:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.html_dir = os.path.join(data_dir, "raw_html")
        os.makedirs(self.html_dir, exist_ok=True)
        self.db = sqlite3.connect(os.path.join(data_dir, "tngt_raw.sqlite3"))
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self.db.commit()

    # --- tien do duyet trang ---
    def get_progress(self, source):
        row = self.db.execute("SELECT * FROM progress WHERE source=?", (source,)).fetchone()
        return dict(row) if row else {"last_page": 0, "finished": 0}

    def set_progress(self, source, page, finished=0, reason=None):
        self.db.execute(
            "INSERT INTO progress(source,last_page,finished,stop_reason,updated_at) "
            "VALUES(?,?,?,?,?) ON CONFLICT(source) DO UPDATE SET last_page=excluded.last_page,"
            " finished=excluded.finished, stop_reason=excluded.stop_reason,"
            " updated_at=excluded.updated_at",
            (source, page, finished, reason, now_iso()))
        self.db.commit()

    # --- URL ---
    def add_urls(self, source, page, links):
        """links: list[(url, title_hint)]. Tra ve danh sach URL moi them."""
        new = []
        for url, hint in links:
            cur = self.db.execute(
                "INSERT OR IGNORE INTO urls(url,source,title_hint,list_page,discovered_at) "
                "VALUES(?,?,?,?,?)", (url, source, hint, page, now_iso()))
            if cur.rowcount:
                new.append(url)
        self.db.commit()
        return new

    def pending_among(self, urls, max_attempts):
        if not urls:
            return []
        q = ",".join("?" * len(urls))
        rows = self.db.execute(
            f"SELECT url FROM urls WHERE url IN ({q}) AND status IN ('pending','error') "
            f"AND attempts < ?", (*urls, max_attempts)).fetchall()
        keep = {r["url"] for r in rows}
        return [u for u in urls if u in keep]  # giu thu tu xuat hien tren trang

    def all_pending(self, sources, max_attempts):
        q = ",".join("?" * len(sources))
        return [dict(r) for r in self.db.execute(
            f"SELECT url, source FROM urls WHERE source IN ({q}) AND status IN ('pending','error') "
            f"AND attempts < ? ORDER BY discovered_at", (*sources, max_attempts))]

    def mark_url(self, url, status, code=None, error=None, count_attempt=True):
        self.db.execute(
            "UPDATE urls SET status=?, http_code=?, error=?, fetched_at=?, "
            "attempts = attempts + ? WHERE url=?",
            (status, code, error, now_iso(), 1 if count_attempt else 0, url))
        self.db.commit()

    def attempts(self, url):
        row = self.db.execute("SELECT attempts FROM urls WHERE url=?", (url,)).fetchone()
        return row["attempts"] if row else 0

    # --- bai viet ---
    def save_html(self, source, url, html):
        folder = os.path.join(self.html_dir, source)
        os.makedirs(folder, exist_ok=True)
        name = hashlib.sha1(url.encode("utf-8")).hexdigest()[:20] + ".html.gz"
        path = os.path.join(folder, name)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(html)
        return os.path.relpath(path, self.data_dir)

    def save_article(self, rec):
        cols = ",".join(rec)
        q = ",".join("?" * len(rec))
        self.db.execute(f"INSERT OR REPLACE INTO articles({cols}) VALUES({q})", tuple(rec.values()))
        self.db.commit()


# =====================================================================
# Boc tach trang danh sach va trang bai viet
# =====================================================================
def extract_links(html, page_url, src):
    art_re = re.compile(src["article_re"])
    exc_re = re.compile(src["exclude_re"]) if src.get("exclude_re") else None
    soup = BeautifulSoup(html, "html.parser")
    seen, out = set(), []
    for a in soup.find_all("a", href=True):
        url = clean_url(urljoin(page_url, a["href"].strip()))
        if url in seen or not art_re.match(url) or (exc_re and exc_re.search(url)):
            continue
        seen.add(url)
        hint = norm_text(a.get("title") or a.get_text(" ", strip=True))[:300]
        out.append((url, hint))
    return out


def _walk_jsonld(obj, found):
    if isinstance(obj, list):
        for x in obj:
            _walk_jsonld(x, found)
    elif isinstance(obj, dict):
        for key in ("datePublished", "headline", "articleSection", "description"):
            if key in obj and key not in found and isinstance(obj[key], (str, list)):
                found[key] = obj[key]
        if "author" in obj and "author" not in found:
            a = obj["author"]
            if isinstance(a, list) and a:
                a = a[0]
            if isinstance(a, dict):
                a = a.get("name")
            if isinstance(a, str):
                found["author"] = a
        for v in obj.values():
            if isinstance(v, (dict, list)):
                _walk_jsonld(v, found)


def extract_meta(html):
    """Lay tieu de, sapo, ngay dang, chuyen muc, tac gia tu the meta va JSON-LD."""
    soup = BeautifulSoup(html, "html.parser")

    def meta(*names):
        for n in names:
            tag = soup.find("meta", attrs={"property": n}) or soup.find("meta", attrs={"name": n}) \
                or soup.find("meta", attrs={"itemprop": n})
            if tag and tag.get("content"):
                return norm_text(tag["content"])
        return None

    ld = {}
    for s in soup.find_all("script", type="application/ld+json"):
        try:
            _walk_jsonld(json.loads(s.string or "", strict=False), ld)
        except (ValueError, TypeError):
            continue
    section = ld.get("articleSection")
    if isinstance(section, list):
        section = ", ".join(map(str, section))
    return {
        "title": meta("og:title") or norm_text(ld.get("headline")) or
        (norm_text(soup.title.string) if soup.title and soup.title.string else None),
        "sapo": meta("og:description", "description") or norm_text(ld.get("description")),
        "published": ld.get("datePublished") or meta("article:published_time", "pubdate",
                                                     "datePublished", "dc.date.issued"),
        "category": norm_text(section) or meta("article:section"),
        "author": norm_text(ld.get("author")) or meta("author", "article:author"),
    }


def extract_article(html, url):
    m = extract_meta(html)
    content, t_meta = None, {}
    if trafilatura is not None:
        out = trafilatura.extract(html, url=url, output_format="json", with_metadata=True,
                                  include_comments=False, include_tables=False,
                                  favor_precision=True)
        if out:
            t_meta = json.loads(out)
            content = norm_text(t_meta.get("text"))
    if not content:  # du phong: noi cac doan <p>
        soup = BeautifulSoup(html, "html.parser")
        paras = [norm_text(p.get_text(" ", strip=True)) for p in soup.find_all("p")]
        content = "\n".join(p for p in paras if len(p) > 40)
    return {
        "title": m["title"] or norm_text(t_meta.get("title")),
        "sapo": m["sapo"] or norm_text(t_meta.get("description")),
        "content": content,
        "published": m["published"] or t_meta.get("date"),
        "category": m["category"] or norm_text(t_meta.get("categories")),
        "author": m["author"] or norm_text(t_meta.get("author")),
    }


CANDIDATE_RE = re.compile("|".join(CANDIDATE_PATTERNS))


def candidate_hits(*texts):
    blob = norm_text(" ".join(t or "" for t in texts)).lower()
    return sorted(set(m.group(0) for m in CANDIDATE_RE.finditer(blob)))


# =====================================================================
# Crawler
# =====================================================================
class Crawler:
    def __init__(self, cfg, store, fetcher, stopper):
        self.cfg = cfg
        self.store = store
        self.fetcher = fetcher
        self.stopper = stopper
        self.stop_before = parse_date(cfg["stop_before"]).date()

    def process_url(self, source, url):
        """Tai + luu 1 bai. Tra ve: 'done' | 'old' | 'error' | 'blocked' | 'skipped'."""
        res = self.fetcher.get(url)
        if res["status"] == "blocked":
            self.store.mark_url(url, "error", res["code"], res["error"], count_attempt=False)
            return "blocked"
        if res["status"] in ("robots", "not_html", "too_large") or res["code"] == 404:
            self.store.mark_url(url, "skipped", res["code"], res["error"])
            return "skipped"
        if res["status"] != "ok":
            failed = self.store.attempts(url) + 1 >= self.cfg["max_attempts_per_url"]
            self.store.mark_url(url, "failed" if failed else "error", res["code"], res["error"])
            return "error"
        try:
            art = extract_article(res["html"], url)
        except Exception as e:  # loi boc tach 1 trang khong duoc lam sap ca qua trinh
            log.exception("Loi boc tach %s", url)
            self.store.mark_url(url, "error", res["code"], f"parse: {e}"[:300])
            return "error"
        html_path = self.store.save_html(source, url, res["html"])
        hits = candidate_hits(art["title"], art["sapo"])
        pub = parse_date(art["published"])
        self.store.save_article({
            "url": url, "source": source, "final_url": res["final_url"],
            "title": art["title"], "sapo": art["sapo"], "content": art["content"],
            "published": pub.isoformat() if pub else art["published"],
            "category": art["category"], "author": art["author"],
            "word_count": len((art["content"] or "").split()),
            "is_candidate": int(bool(hits)), "candidate_hits": ", ".join(hits),
            "html_path": html_path, "fetched_at": now_iso(),
        })
        self.stopper.articles_done += 1
        is_old = pub is not None and pub.date() < self.stop_before
        self.store.mark_url(url, "old" if is_old else "done", res["code"])
        return "old" if is_old else "done"

    def _handle_block(self, name, blocks):
        wait = self.cfg["block_cooldown_seconds"] * blocks
        log.warning("[%s] bi chan (lan %d), nghi %ds", name, blocks, wait)
        self.stopper.sleep(wait)

    def run_source(self, key, restart=False, max_pages=None):
        src, cfg = SOURCES[key], self.cfg
        name = src["name"]
        max_pages = max_pages or cfg["max_pages_per_source"]
        prog = self.store.get_progress(key)
        # Lan truoc dung giua chung -> chay tiep. Da xong hoac --restart -> lam lai tu trang 1
        # (link da co trong CSDL se duoc bo qua, het link moi thi tu dung).
        start = 1 if (restart or prog["finished"] or prog["last_page"] == 0) else prog["last_page"] + 1
        log.info("=== %s: bat dau tu trang %d (toi da %d trang) ===", name, start, max_pages)

        empty_pages = old_streak = consec_err = blocks = 0
        prev_links = None
        stop_reason, page = None, start - 1

        for page in range(start, max_pages + 1):
            if self.stopper.should_stop():
                stop_reason = None  # dung toan cuc, khong danh dau nguon la xong
                break
            page_url = src["page_url"](page)
            res = self.fetcher.get(page_url)

            if res["status"] == "blocked":
                blocks += 1
                if blocks > cfg["max_blocks_per_source"]:
                    stop_reason = f"bi chan qua {cfg['max_blocks_per_source']} lan"
                    break
                self._handle_block(name, blocks)
                continue
            if res["status"] != "ok":
                if res["code"] == 404:
                    stop_reason = f"trang {page} tra ve 404 (het phan trang)"
                    break
                consec_err += 1
                log.warning("[%s] trang %d loi: %s", name, page, res["error"])
                if consec_err >= cfg["max_consecutive_errors"]:
                    stop_reason = f"{consec_err} loi lien tiep"
                    break
                continue

            links = extract_links(res["html"], page_url, src)[: cfg["max_links_per_page"]]
            link_set = {u for u, _ in links}
            if not links:
                stop_reason = f"trang {page} khong co link bai viet (het phan trang / doi giao dien?)"
                break
            if prev_links is not None and link_set == prev_links:
                stop_reason = f"trang {page} giong het trang truoc (site tra lai trang cu)"
                break
            prev_links = link_set

            new = self.store.add_urls(key, page, links)
            todo = self.store.pending_among([u for u, _ in links], cfg["max_attempts_per_url"])
            log.info("[%s] trang %d: %d link, %d moi, %d can tai",
                     name, page, len(links), len(new), len(todo))
            empty_pages = 0 if new else empty_pages + 1

            for url in todo:
                if self.stopper.should_stop():
                    break
                r = self.process_url(key, url)
                if r == "blocked":
                    blocks += 1
                    if blocks > cfg["max_blocks_per_source"]:
                        stop_reason = f"bi chan qua {cfg['max_blocks_per_source']} lan"
                        break
                    self._handle_block(name, blocks)
                    continue
                consec_err = consec_err + 1 if r == "error" else 0
                if r == "old":
                    old_streak += 1
                elif r == "done":
                    old_streak = 0
                if consec_err >= cfg["max_consecutive_errors"]:
                    stop_reason = f"{consec_err} loi lien tiep khi tai bai"
                    break
                if old_streak >= cfg["old_streak_to_stop"]:
                    stop_reason = f"{old_streak} bai lien tiep cu hon {cfg['stop_before']}"
                    break

            # chi ghi trang da xong khi da duyet het bai cua trang do
            if not self.stopper.flag:
                self.store.set_progress(key, page)
            elif not stop_reason:
                break  # dung toan cuc giua trang: khong roi vao nhanh else (danh dau xong)
            if stop_reason:
                break
            if empty_pages >= cfg["empty_pages_to_stop"]:
                stop_reason = f"{empty_pages} trang lien tiep khong co link moi"
                break
        else:
            stop_reason = f"dat gioi han {max_pages} trang"

        if stop_reason:
            self.store.set_progress(key, page, finished=1, reason=stop_reason)
            log.info("=== %s: dung nguon - %s ===", name, stop_reason)
        else:
            log.info("=== %s: tam dung (%s), lan sau chay tiep ===", name, self.stopper.reason)

    def fetch_pending(self, sources):
        rows = self.store.all_pending(sources, self.cfg["max_attempts_per_url"])
        log.info("Co %d bai pending/loi can tai lai", len(rows))
        blocks = 0
        for row in rows:
            if self.stopper.should_stop():
                break
            r = self.process_url(row["source"], row["url"])
            if r == "blocked":
                blocks += 1
                if blocks > self.cfg["max_blocks_per_source"]:
                    log.warning("Bi chan qua nhieu lan, dung.")
                    break
                self._handle_block(row["source"], blocks)


# =====================================================================
# Thong ke va xuat CSV
# =====================================================================
def print_stats(store):
    print("\n--- Tien do duyet trang ---")
    for r in store.db.execute("SELECT * FROM progress ORDER BY source"):
        print(f"{r['source']:<11} trang {r['last_page']:<4} "
              f"{'XONG' if r['finished'] else 'dang do'}  {r['stop_reason'] or ''}")
    print("\n--- Trang thai URL ---")
    for r in store.db.execute(
            "SELECT source, status, COUNT(*) n FROM urls GROUP BY source, status ORDER BY source"):
        print(f"{r['source']:<11} {r['status']:<8} {r['n']}")
    print("\n--- Bai viet da luu ---")
    for r in store.db.execute(
            "SELECT source, COUNT(*) n, SUM(is_candidate) c, MIN(published) a, MAX(published) b "
            "FROM articles GROUP BY source"):
        print(f"{r['source']:<11} {r['n']:>6} bai | {r['c'] or 0:>6} co tu khoa TNGT | "
              f"{(r['a'] or '')[:10]} -> {(r['b'] or '')[:10]}")
    total = store.db.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    print(f"\nTong: {total} bai\n")


def export_csv(store, out_path, only_candidates=False):
    cols = ["source", "url", "published", "title", "sapo", "content", "category", "author",
            "word_count", "is_candidate", "candidate_hits", "html_path", "fetched_at"]
    q = f"SELECT {','.join(cols)} FROM articles"
    if only_candidates:
        q += " WHERE is_candidate=1"
    q += " ORDER BY published DESC"
    n = 0
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(cols)
        for row in store.db.execute(q):  # doc tung dong, khong nap het vao RAM
            w.writerow(list(row))
            n += 1
    log.info("Da xuat %d bai ra %s", n, out_path)


# =====================================================================
# Main
# =====================================================================
def main():
    ap = argparse.ArgumentParser(description="Cao du lieu tho TNGT duong bo theo path")
    ap.add_argument("command", nargs="?", default="run", choices=["run", "fetch", "stats", "export"],
                    help="mac dinh: run (khi bam Run trong IDE khong co tham so)")
    ap.add_argument("--sources", default=",".join(SOURCES),
                    help=f"cac nguon, cach nhau dau phay ({','.join(SOURCES)})")
    ap.add_argument("--max-pages", type=int, help="ghi de max_pages_per_source")
    ap.add_argument("--max-articles", type=int, help="ghi de max_articles_per_run")
    ap.add_argument("--max-minutes", type=int, help="ghi de max_runtime_minutes")
    ap.add_argument("--stop-before", help="moc ngay YYYY-MM-DD, ghi de stop_before")
    ap.add_argument("--restart", action="store_true", help="duyet lai tu trang 1")
    ap.add_argument("--out", default=None, help="duong dan file CSV khi export")
    ap.add_argument("--only-candidates", action="store_true",
                    help="export chi bai co tu khoa TNGT")
    args = ap.parse_args()

    cfg = dict(CONFIG)
    if args.max_articles:
        cfg["max_articles_per_run"] = args.max_articles
    if args.max_minutes:
        cfg["max_runtime_minutes"] = args.max_minutes
    if args.stop_before:
        cfg["stop_before"] = args.stop_before

    setup_logging(cfg["data_dir"])
    store = Store(cfg["data_dir"])

    if args.command == "stats":
        print_stats(store)
        return
    if args.command == "export":
        export_csv(store, args.out or os.path.join(cfg["data_dir"], "raw_articles.csv"),
                   args.only_candidates)
        return

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = [s for s in sources if s not in SOURCES]
    if unknown:
        ap.error(f"nguon khong hop le: {unknown}")
    if trafilatura is None:
        log.warning("Chua cai trafilatura: se dung cach noi <p> don gian (pip install trafilatura)")

    stopper = StopController(cfg)
    stopper.install_signal()
    crawler = Crawler(cfg, store, Fetcher(cfg, stopper), stopper)
    try:
        if args.command == "run":
            for key in sources:
                if stopper.should_stop():
                    break
                crawler.run_source(key, restart=args.restart, max_pages=args.max_pages)
        elif args.command == "fetch":
            crawler.fetch_pending(sources)
    except KeyboardInterrupt:
        log.warning("Da thoat ngay theo yeu cau. Du lieu da ghi truoc do van an toan.")
    finally:
        mins = (time.time() - stopper.started) / 60
        log.info("Ket thuc sau %.1f phut, tai %d bai moi. %s", mins, stopper.articles_done,
                 f"Ly do dung: {stopper.reason}" if stopper.reason else "")
        print_stats(store)
        store.db.close()


if __name__ == "__main__":
    main()