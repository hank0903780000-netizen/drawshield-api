from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
import fitz
import stripe
import uuid, os, asyncio, io
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

try:
    import pytesseract
    TESSERACT_OK = True
except ImportError:
    TESSERACT_OK = False

# PaddleOCR（透過 RapidOCR 跑 PP-OCR 模型）：中文公司名/客戶名辨識遠優於 Tesseract
_RAPID_OCR = None
try:
    from rapidocr_onnxruntime import RapidOCR
    RAPIDOCR_OK = True
except ImportError:
    RAPIDOCR_OK = False

def get_rapidocr():
    global _RAPID_OCR
    if _RAPID_OCR is None and RAPIDOCR_OK:
        _RAPID_OCR = RapidOCR()
    return _RAPID_OCR

app = FastAPI()

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "https://drawshield.vercel.app")

PRICE_CENTS = {"rotate": 300, "redact": 300, "full": 500}
# 綠界 ECPay 為新台幣計價（信用卡）
PRICE_TWD = {"rotate": 100, "redact": 100, "full": 150}
SERVICE_NAMES = {
    "rotate": "PDF 旋轉",
    "redact": "遮蔽公司名稱和 LOGO",
    "full": "旋轉 + 遮蔽公司名稱和 LOGO",
}

# ── 綠界 ECPay 全方位金流設定 ─────────────────────────────────────────────
# 預設為綠界官方「測試環境」公開測試帳號，正式上線時於 Railway 設定環境變數覆蓋
ECPAY_MERCHANT_ID = os.environ.get("ECPAY_MERCHANT_ID", "2000132")
ECPAY_HASH_KEY = os.environ.get("ECPAY_HASH_KEY", "5294y06JbISpM5x9")
ECPAY_HASH_IV = os.environ.get("ECPAY_HASH_IV", "v77hoKGq4kWxNNIS")
ECPAY_ENV = os.environ.get("ECPAY_ENV", "stage")  # "stage" 或 "production"
ECPAY_AIO_URL = (
    "https://payment.ecpay.com.tw/Cgi-Bin/AioCheckOut/V5"
    if ECPAY_ENV == "production"
    else "https://payment-stage.ecpay.com.tw/Cgi-Bin/AioCheckOut/V5"
)
# 後端公開網址（綠界 server-to-server 回呼 ReturnURL 用）
API_PUBLIC_URL = os.environ.get(
    "API_PUBLIC_URL",
    "https://enchanting-integrity-production-36f6.up.railway.app",
)

# 訂單暫存（單機記憶體即可；檔案 30 分鐘後自動刪，重啟遺失可接受）
ECPAY_ORDERS = {}  # MerchantTradeNo -> {upload_ids, service, rotate_deg, company_name, paid, results}

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("/tmp/drawshield")
UPLOAD_DIR.mkdir(exist_ok=True)
VERSION = "paddleocr-integrated"


async def auto_delete(path: str, delay: int = 60):
    await asyncio.sleep(delay)
    try:
        os.remove(path)
    except Exception:
        pass


def remove_watermark_artifacts(doc):
    """移除標準 PDF 浮水印（/Artifact /Watermark 標記內容區塊）。"""
    import re
    pat = re.compile(rb"/Artifact\s*<<[^>]*?/Watermark[^>]*?>>\s*BDC.*?EMC\s*", re.DOTALL)
    for page in doc:
        for xref in page.get_contents():
            stream = doc.xref_stream(xref)
            if stream and b"/Watermark" in stream:
                new = pat.sub(b"", stream)
                if new != stream:
                    doc.update_stream(xref, new)


def apply_text_redaction(doc, company_name: str) -> bool:
    """Returns True if any redactions were applied (text found and whited out)."""
    import re
    names = [n.strip() for n in company_name.split(",") if n.strip()]
    phone_pattern = re.compile(r"(?:TEL|FAX|AX|電話|傳真)\s*[:：]?\s*[\d\-\+\(\) ]{5,}", re.IGNORECASE)
    # Auto-detect English company name pattern (reliable on Linux despite CJK surrogate issue)
    english_co_pattern = re.compile(
        r'\b[A-Z][A-Z\s\.\-]{1,}\s+(?:CORP(?:ORATION)?\.?|INC\.?|LTD\.?|LIMITED|COMPANY|GMBH|CO\.,?\s*LTD\.?)\b',
        re.IGNORECASE
    )
    # 所有權/機密聲明（整個 span 直接遮蔽，不限位置）
    ownership_pattern = re.compile(
        r'智慧財產|機密文件|集團所有|公司所有|CONFIDENTIAL|PROPRIETARY|ALL RIGHTS RESERVED',
        re.IGNORECASE
    )
    # 全頁掃描用：縮寫電話（T:/F: 接號碼）與完整英文公司名（含 ENGINEERING 等）
    phone_abbr_pattern = re.compile(
        r'(?:TEL|FAX|電話|傳真|电话|传真|\bT|\bF)\s*[:：]\s*[\d\-\+\(\)]{5,}',
        re.IGNORECASE
    )

    any_redacted_doc = False
    for page in doc:
        # 浮水印移除：淡灰色文字（不分內容/位置）一律遮蔽。
        # 工程圖標注為黑/藍/紅深色，淡色大字必為浮水印
        # 用 rawdict 取得逐字元框：斜放浮水印的 span bbox 是巨大方框，
        # 整框塗白會蓋掉圖面；改為逐字元刪除字形且不塗色（fill=False）
        for b in page.get_text("rawdict")["blocks"]:
            if b["type"] != 0:
                continue
            for line in b["lines"]:
                for span in line["spans"]:
                    col = span.get("color", 0)
                    r = (col >> 16) & 255
                    g = (col >> 8) & 255
                    bl = col & 255
                    clean_sp = ''.join(
                        ch["c"] for ch in span["chars"]
                        if ord(ch["c"]) < 0xD800 or ord(ch["c"]) > 0xDFFF
                    )
                    if (min(r, g, bl) > 150 or ownership_pattern.search(clean_sp)
                            or clean_sp.strip() in ("密", "機密", "极密", "極密")):
                        for ch in span["chars"]:
                            page.add_redact_annot(fitz.Rect(ch["bbox"]), fill=False)
                    # 全頁掃描縮寫電話 / 完整英文公司名（不限窄帶，整 span 塗白）
                    elif (phone_abbr_pattern.search(clean_sp)
                            or english_co_pattern.search(clean_sp)):
                        page.add_redact_annot(fitz.Rect(span["bbox"]), fill=(1, 1, 1))
        # 「機 密」拆成兩個相鄰 span 的戳記：機+密 距離近時一併遮蔽
        mi_spans, ji_spans = [], []
        for b in page.get_text("dict")["blocks"]:
            if b["type"] != 0:
                continue
            for line in b["lines"]:
                for span in line["spans"]:
                    t = ''.join(c for c in span["text"] if ord(c) < 0xD800 or ord(c) > 0xDFFF).strip()
                    if t == "機":
                        ji_spans.append(span["bbox"])
                    elif t == "密":
                        mi_spans.append(span["bbox"])
        for jb in ji_spans:
            for mb_ in mi_spans:
                dx = max(jb[0], mb_[0]) - min(jb[2], mb_[2])
                dy = max(jb[1], mb_[1]) - min(jb[3], mb_[3])
                if dx < 30 and dy < 30:
                    page.add_redact_annot(fitz.Rect(jb), fill=(1, 1, 1))
                    page.add_redact_annot(fitz.Rect(mb_), fill=(1, 1, 1))
        # Use mediabox (unrotated) dimensions — text coords are always in media space
        mb = page.mediabox
        mw, mh = mb.width, mb.height
        # Title block: narrow edge strips in media space (10% of shorter side or bottom 10%)
        STRIP = min(mw, mh) * 0.10
        # Collect title block spans: narrow left/right/top/bottom strip, but skip
        # spans clearly in the drawing body (e.g. top-half of a narrow left strip)
        all_spans = []
        for b in page.get_text("dict")["blocks"]:
            if b["type"] != 0:
                continue
            for line in b["lines"]:
                for span in line["spans"]:
                    x0, y0, x1, y1 = span["bbox"]
                    in_strip = (x0 < STRIP or x1 > mw - STRIP or
                                y0 < STRIP or y1 > mh - STRIP)
                    if not in_strip:
                        continue
                    # Skip spans in the upper half of a narrow left/right strip
                    # (those are NOTE/annotation text, not title block)
                    if (x0 < STRIP or x1 > mw - STRIP) and y1 < mh * 0.45:
                        continue
                    all_spans.append(span)

        def clean_str(s: str) -> str:
            """Remove surrogate escapes (Linux PyMuPDF encoding artifact)."""
            return ''.join(c for c in s if ord(c) < 0xD800 or ord(c) > 0xDFFF)

        def text_matches(span_text: str, name: str) -> bool:
            """Check if span_text matches name — handles surrogates and partial streams."""
            st = clean_str(span_text).strip()
            n = clean_str(name).strip()
            if not st or not n:
                return False
            if st in n or n in st:
                return True
            # 任意 2 個以上連續字元重疊（CJK 短字串也能比對）
            min_len = min(len(st), len(n), 2)
            if min_len < 2:
                return False
            for i in range(len(n) - min_len + 1):
                if n[i:i + min_len] in st:
                    return True
            return False

        def rect_contains_span(rect, span_bbox):
            """True if rect overlaps significantly with span_bbox (search_for may return partial rect)."""
            sx0, sy0, sx1, sy1 = span_bbox
            # x 重疊 且 rect 的 y0 在 span 的 y 範圍內（部分匹配也算）
            x_overlap = rect.x0 <= sx1 + 2 and rect.x1 >= sx0 - 2
            y_overlap = rect.y0 <= sy1 and rect.y1 >= sy0
            return x_overlap and y_overlap

        # Auto mode: no company name → try to detect English company name pattern
        # (CJK text has surrogate escapes on Linux so English detection is more reliable)
        auto_cluster_rects = []  # CJK 公司名群集（自動模式直接遮蔽的 bbox）
        if not names:
            for span in all_spans:
                clean_t = ''.join(c for c in span["text"] if ord(c) < 0xD800 or ord(c) > 0xDFFF).strip()
                if english_co_pattern.search(clean_t):
                    names.append(clean_t)
                    break
            # CJK 公司名偵測：常見每字一個 span（直排/橫排），
            # 先把相鄰 span 群組成字串鏈，再比對「有限公司」等字尾。
            # 掃整頁（公司名欄位不一定在邊緣帶內），字尾比對精確不會誤殺尺寸標注
            cjk_suffix = re.compile(r"(?:股份)?有限公司|株式会社|实业|實業")
            spans_left = []
            for b in page.get_text("dict")["blocks"]:
                if b["type"] != 0:
                    continue
                for line in b["lines"]:
                    spans_left.extend(line["spans"])
            clusters = []
            used = [False] * len(spans_left)
            GAP = 4  # 相鄰 span 最大間距 (pt)；title block 欄位間距約 7pt，需小於此值
            for i, s in enumerate(spans_left):
                if used[i]:
                    continue
                chain = [i]
                used[i] = True
                grew = True
                while grew:
                    grew = False
                    for j, s2 in enumerate(spans_left):
                        if used[j]:
                            continue
                        b2 = s2["bbox"]
                        for k in chain:
                            b1 = spans_left[k]["bbox"]
                            # x 範圍重疊（同欄直排）且 y 距離小，或 y 重疊（同列橫排）且 x 距離小
                            x_ov = b1[0] < b2[2] and b1[2] > b2[0]
                            y_ov = b1[1] < b2[3] and b1[3] > b2[1]
                            y_gap = max(b1[1], b2[1]) - min(b1[3], b2[3])
                            x_gap = max(b1[0], b2[0]) - min(b1[2], b2[2])
                            if (x_ov and y_gap < GAP) or (y_ov and x_gap < GAP):
                                chain.append(j)
                                used[j] = True
                                grew = True
                                break
                clusters.append(chain)
            for chain in clusters:
                # 依幾何順序串接文字（直排由上而下 / 橫排由左而右皆涵蓋；
                # 旋轉頁面直排可能 y 遞減，因此兩種順序都試）
                chs = [spans_left[k] for k in chain]
                t_fwd = ''.join(clean_str(c["text"]) for c in sorted(chs, key=lambda c: (c["bbox"][1], c["bbox"][0]))).replace(" ", "")
                t_rev = ''.join(clean_str(c["text"]) for c in sorted(chs, key=lambda c: (-c["bbox"][1], c["bbox"][0]))).replace(" ", "")
                if cjk_suffix.search(t_fwd) or cjk_suffix.search(t_rev):
                    xs = [c["bbox"][0] for c in chs] + [c["bbox"][2] for c in chs]
                    ys = [c["bbox"][1] for c in chs] + [c["bbox"][3] for c in chs]
                    auto_cluster_rects.append(fitz.Rect(min(xs), min(ys), max(xs), max(ys)))

        # 找到匹配的 span，並擴展遮蔽同一欄位（相同 x 範圍）的所有文字
        matched_x_bands = []  # [(x0, x1, y0, y1)] 已匹配的欄位範圍
        any_redacted = False
        for rect in auto_cluster_rects:
            page.add_redact_annot(rect, fill=(1, 1, 1))
            any_redacted = True
        for name in names:
            # 方法一：search_for（英文/簡單文字效果好）
            rects = page.search_for(name)
            for rect in rects:
                span_found = False
                for span in all_spans:
                    if rect_contains_span(rect, span["bbox"]):
                        full_rect = fitz.Rect(span["bbox"])
                        page.add_redact_annot(full_rect, fill=(1, 1, 1))
                        matched_x_bands.append((full_rect.x0 - 5, full_rect.x1 + 5, full_rect.y0, full_rect.y1))
                        span_found = True
                        any_redacted = True
                if not span_found:
                    page.add_redact_annot(rect, fill=(1, 1, 1))
                    matched_x_bands.append((rect.x0 - 5, rect.x1 + 5, rect.y0, rect.y1))
                    any_redacted = True
            # 方法二：span 文字比對（處理 search_for 完全找不到的情況）
            if not rects:
                for span in all_spans:
                    if text_matches(span["text"], name):
                        full_rect = fitz.Rect(span["bbox"])
                        page.add_redact_annot(full_rect, fill=(1, 1, 1))
                        matched_x_bands.append((full_rect.x0 - 5, full_rect.x1 + 5, full_rect.y0, full_rect.y1))
                        any_redacted = True

        # 遮蔽同一欄位內所有其他文字（英文名稱等）：x 和 y 都必須重疊
        for span in all_spans:
            sx0, sy0, sx1, sy1 = span["bbox"]
            for bx0, bx1, by0, by1 in matched_x_bands:
                # span 的 x 範圍與匹配欄位重疊，且 y 範圍也重疊
                if sx0 <= bx1 and sx1 >= bx0 and sy0 <= by1 and sy1 >= by0:
                    page.add_redact_annot(fitz.Rect(span["bbox"]), fill=(1, 1, 1))
                    break

        # 自動遮蔽 TEL / FAX 電話號碼（title block 區域）
        for span in all_spans:
            if phone_pattern.search(span["text"]):
                page.add_redact_annot(fitz.Rect(span["bbox"]), fill=(1, 1, 1))
                any_redacted = True

        page.apply_redactions()
        if any_redacted:
            any_redacted_doc = True

    return any_redacted_doc


def apply_logo_redaction(doc):
    for page in doc:
        # Use mediabox for area/edge calculations (path coords are in media space)
        mb = page.mediabox
        page_area = mb.width * mb.height
        redacted = False

        # ── 嵌入式點陣圖 ────────────────────────────────────────────────
        img_list = page.get_images(full=True)
        for img in img_list:
            xref = img[0]
            try:
                rects = page.get_image_rects(xref)
            except Exception:
                continue
            for rect in rects:
                img_area = rect.width * rect.height
                if page_area > 0 and img_area / page_area > 0.5:
                    continue
                page.add_redact_annot(rect, fill=(1, 1, 1))
                redacted = True

        # ── 向量 LOGO：找 title block 欄位內的有色路徑 ──────────────────
        # 策略：title block 通常是圖面邊緣的窄欄/窄列。
        # 工程圖標注（尺寸線）分布在主圖區；title block 的有色路徑（LOGO）
        # 通常集中在某個小區域，與主圖區有明顯距離。
        # 做法：找所有「完全不在主圖區」的有色路徑群集。
        drawings = page.get_drawings()

        # 先估算主圖區邊界（文字的主要分布區域，排除 title block 文字）
        # 方法：找最密集的文字 x/y 範圍
        all_text = []
        for b in page.get_text("dict")["blocks"]:
            if b["type"] == 0:
                all_text.append(b["bbox"])

        # 收集有色路徑，按「群集」分組
        colored_paths = []
        for d in drawings:
            c = d.get("color") or d.get("fill")
            if c is None:
                continue
            r, g, b = c[0], c[1], c[2]
            if (r < 0.1 and g < 0.1 and b < 0.1) or (r > 0.9 and g > 0.9 and b > 0.9):
                continue
            dr = fitz.Rect(d["rect"])
            area = dr.width * dr.height
            if page_area > 0 and area / page_area > 0.3:
                continue
            colored_paths.append(dr)

        # 用 DBSCAN 概念：把距離相近的有色路徑合併成群
        CLUSTER_DIST = 40  # pt，同一群的最大距離
        groups = []
        used = set()
        for i, p in enumerate(colored_paths):
            if i in used:
                continue
            group = [p]
            used.add(i)
            for j, q in enumerate(colored_paths):
                if j in used:
                    continue
                # 距離 = 兩個 rect 的 center 距離
                cx1 = (p.x0 + p.x1) / 2
                cy1 = (p.y0 + p.y1) / 2
                cx2 = (q.x0 + q.x1) / 2
                cy2 = (q.y0 + q.y1) / 2
                if abs(cx1 - cx2) < CLUSTER_DIST and abs(cy1 - cy2) < CLUSTER_DIST:
                    group.append(q)
                    used.add(j)
            groups.append(group)

        # 找面積小且位於頁面邊緣（title block 區）的群集作為 LOGO 候選
        pw = mb.width   # media box dimensions (same space as path coords)
        ph = mb.height
        EDGE = 0.18  # 邊緣 18% 範圍視為 title block 區域
        logo_annots = []
        for group in groups:
            xs2 = [r.x0 for r in group] + [r.x1 for r in group]
            ys2 = [r.y0 for r in group] + [r.y1 for r in group]
            merged = fitz.Rect(min(xs2), min(ys2), max(xs2), max(ys2))
            area = merged.width * merged.height
            if page_area > 0 and area / page_area >= 0.02:
                continue  # 太大，是主圖標注群
            # 必須位於頁面任意一側邊緣（title block 所在位置）
            in_edge = (
                merged.x1 < pw * EDGE or merged.x0 > pw * (1 - EDGE) or
                merged.y1 < ph * EDGE or merged.y0 > ph * (1 - EDGE)
            )
            if in_edge:
                logo_annots.append(merged)

        for rect in logo_annots:
            page.add_redact_annot(rect, fill=(1, 1, 1))
            redacted = True

        if redacted:
            page.apply_redactions()


def is_scanned_page(page) -> bool:
    """頁面是否為純掃描圖片（無文字層、只有一張全頁圖片）。"""
    if page.get_text().strip():
        return False
    imgs = page.get_images(full=True)
    if len(imgs) == 0:
        return False
    try:
        page_area = page.rect.width * page.rect.height
        for img in imgs:
            rects = page.get_image_rects(img[0])
            for r in rects:
                if page_area > 0 and r.width * r.height / page_area > 0.7:
                    return True
    except Exception:
        pass
    return False


SENSITIVE_LINE = None  # lazy-compiled

def _sensitive_pattern():
    global SENSITIVE_LINE
    if SENSITIVE_LINE is None:
        import re
        SENSITIVE_LINE = re.compile(
            r"公司|集團|集团|企業社|株式会社|实业|實業|"
            r"CO\W{0,2}LTD|CORP|INC\b|\bLTD\b|LIMITED|GMBH|"
            r"ENGINEERING|INDUSTR|TECHNOLOG|ENTERPRISE|"
            r"TEL|FAX|電話|傳真|电话|传真|"
            r"\bT\s*[:：]\s*\d|\bF\s*[:：]\s*\d|"  # T:/F: 縮寫後接數字
            r"機密|机密|CONFIDENTIAL|PROPRIETARY",
            re.IGNORECASE,
        )
    return SENSITIVE_LINE


COMPANY_LINE = None

def _company_pattern():
    global COMPANY_LINE
    if COMPANY_LINE is None:
        import re
        COMPANY_LINE = re.compile(
            r"公司|集團|集团|企業社|株式会社|实业|實業|"
            r"CO\W{0,2}LTD|CORP|INC\b|\bLTD\b|LIMITED|GMBH|"
            r"ENGINEERING|INDUSTR|TECHNOLOG|ENTERPRISE",
            re.IGNORECASE,
        )
    return COMPANY_LINE


def _expand_to_cell(gray, box):
    """把像素框沿表格黑色格線擴張到整個儲存格（找不到格線則维持原框）。"""
    h, w = gray.shape
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, w - 1), min(y1, h - 1)
    LIM_X, LIM_Y = int(w * 0.30), int(h * 0.08)
    bh = max(y1 - y0, 8)
    # 直格線需貫穿超出文字行高的範圍（字的筆畫只在行內，不會誤判）
    vy0, vy1 = max(0, y0 - bh), min(h - 1, y1 + bh)
    def v_line(x):
        col = gray[vy0:vy1, x]
        return col.size and (col < 160).mean() > 0.45
    def h_line(y):
        row = gray[y, x0:x1]
        return row.size and (row < 160).mean() > 0.45
    # 真格線判別：細線(≤5px) 且其後 18px 內全白（格線旁是儲存格留白）；
    # LOGO/文字筆畫後方近距離內還有黑色內容 → 視為內容穿過並納入遮蔽
    def col_has_ink(x):
        col = gray[vy0:vy1, x]
        return col.size and (col < 160).mean() > 0.15
    def run_width(x, step):
        n = 0
        while 0 <= x < w and v_line(x) and n < 60:
            n += 1
            x += step
        return n
    def is_border(x, rw, step):
        if rw > 12:
            return False
        probe0 = x + step * rw
        for k in range(1, 19):
            xx = probe0 + step * k
            if 0 <= xx < w and col_has_ink(xx):
                return False
        return True
    def scan(x_start, step, lim):
        x = x_start
        while (x > lim) if step < 0 else (x < lim):
            if v_line(x):
                rw = run_width(x, step)
                if is_border(x, rw, step):
                    return x - step  # 停在格線內側
                x += step * max(rw, 1)
            else:
                x += step
        return None
    res = scan(x0, -1, max(0, x0 - LIM_X))
    nx0 = res if res is not None else x0
    res = scan(x1, 1, min(w - 1, x1 + LIM_X))
    nx1 = res if res is not None else x1

    # 上下邊框：用「最長連續暗點」橫貫整格寬度判別（真邊框連續，文字有斷點）
    def longest_dark_run(arr_row):
        best = run = 0
        for v in arr_row:
            if v < 160:
                run += 1
                if run > best:
                    best = run
            else:
                run = 0
        return best
    span = max(nx1 - nx0, 1)
    need = span * 0.6  # 連續暗點需達整格寬 60% 才算邊框
    ny0 = y0
    for y in range(y0, max(0, y0 - LIM_Y) - 1, -1):
        if longest_dark_run(gray[y, nx0:nx1]) >= need:
            ny0 = y + 1
            break
        ny0 = y
    ny1 = y1
    for y in range(y1, min(h - 1, y1 + LIM_Y) + 1):
        if longest_dark_run(gray[y, nx0:nx1]) >= need:
            ny1 = y - 1
            break
        ny1 = y
    return (nx0, ny0, nx1, ny1)


def _tight_company_box(gray, box):
    """緊貼公司名區塊（商標+中文名+英文名）：從英文名框沿四向長大，
    遇到「整段空白」即停，不撐到格線、不吃鄰格。"""
    h, w = gray.shape
    x0, y0, x1, y1 = [int(v) for v in box]
    x0 = max(x0, 0); y0 = max(y0, 0); x1 = min(x1, w - 1); y1 = min(y1, h - 1)
    bh = max(y1 - y0, 8)
    GAP = max(int(bh * 0.9), 6)   # 視為「區塊邊界」的連續空白量
    LIM = int(bh * 4)             # 各方向最大延伸
    DARK = 140

    def col_has_ink(x, ylo, yhi):
        c = gray[max(0, ylo):min(h, yhi), x]
        return c.size and int((c < DARK).sum()) >= 2

    def row_has_ink(y, xlo, xhi):
        r = gray[y, max(0, xlo):min(w, xhi)]
        return r.size and int((r < DARK).sum()) >= 2

    # 先上下長大（涵蓋英文名上方的中文名），用目前寬度判斷
    ny0 = y0
    gap = 0
    for y in range(y0 - 1, max(0, y0 - LIM) - 1, -1):
        if row_has_ink(y, x0, x1):
            ny0 = y; gap = 0
        else:
            gap += 1
            if gap >= GAP:
                break
    ny1 = y1
    gap = 0
    for y in range(y1 + 1, min(h - 1, y1 + LIM) + 1):
        if row_has_ink(y, x0, x1):
            ny1 = y; gap = 0
        else:
            gap += 1
            if gap >= GAP:
                break
    # 再左右長大（涵蓋商標），用已長好的高度判斷
    nx0 = x0
    gap = 0
    for x in range(x0 - 1, max(0, x0 - LIM) - 1, -1):
        if col_has_ink(x, ny0, ny1):
            nx0 = x; gap = 0
        else:
            gap += 1
            if gap >= GAP:
                break
    nx1 = x1
    gap = 0
    for x in range(x1 + 1, min(w - 1, x1 + LIM) + 1):
        if col_has_ink(x, ny0, ny1):
            nx1 = x; gap = 0
        else:
            gap += 1
            if gap >= GAP:
                break
    return (nx0 - 2, ny0 - 2, nx1 + 2, ny1 + 2)


def ocr_sensitive_boxes(img, expand=True):
    """整頁 + 底部標題欄區各 OCR 一次（局部 OCR 對標題欄小字辨識率較高），
    合併敏感行框。"""
    boxes = _ocr_sensitive_boxes_single(img, expand)
    if boxes is None:
        return None
    w, h = img.size
    y_off = int(h * 0.70)
    crop = img.crop((0, y_off, w, h))
    for psm in (11, 6):  # 稀疏 + 整齊版面兩種模式，提高標題欄辨識率
        extra = _ocr_sensitive_boxes_single(crop, expand, psm=psm) or []
        for (x0, y0, x1, y1) in extra:
            boxes.append((x0, y0 + y_off, x1, y1 + y_off))
    # 合併重疊框
    merged = []
    for bx in boxes:
        for i, mb_ in enumerate(merged):
            if bx[0] < mb_[2] and bx[2] > mb_[0] and bx[1] < mb_[3] and bx[3] > mb_[1]:
                merged[i] = (min(bx[0], mb_[0]), min(bx[1], mb_[1]),
                             max(bx[2], mb_[2]), max(bx[3], mb_[3]))
                break
        else:
            merged.append(bx)
    return merged


def _ocr_sensitive_boxes_single(img, expand=True, psm=11):
    """OCR 影像，回傳含公司名/電話/機密字樣之「行」的像素框 [(x0,y0,x1,y1)]。
    expand=True 時公司名行沿表格格線擴張至整個儲存格。"""
    if not TESSERACT_OK:
        return None  # OCR 不可用
    try:
        data = pytesseract.image_to_data(
            img, lang="chi_tra+eng",
            output_type=pytesseract.Output.DICT,
            config=f"--psm {psm}",
        )
    except Exception:
        return None
    pat = _sensitive_pattern()
    co_pat = _company_pattern()
    gray = np.array(img.convert("L"))
    # 依 (block, par, line) 分行
    lines = {}
    for i, token in enumerate(data["text"]):
        if not token.strip() or int(data["conf"][i]) < 10:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(i)
    boxes = []
    for key, idxs in lines.items():
        joined = "".join(data["text"][i] for i in idxs)
        if pat.search(joined):
            x0 = min(data["left"][i] for i in idxs)
            y0 = min(data["top"][i] for i in idxs)
            x1 = max(data["left"][i] + data["width"][i] for i in idxs)
            y1 = max(data["top"][i] + data["height"][i] for i in idxs)
            img_w = img.size[0]
            if (x1 - x0) > img_w * 0.5:
                # OCR 把整列當一行（常見於 psm 6）：只取個別命中的 token
                for i in idxs:
                    tk = data["text"][i]
                    if pat.search(tk):
                        bx = (data["left"][i] - 4, data["top"][i] - 4,
                              data["left"][i] + data["width"][i] + 4,
                              data["top"][i] + data["height"][i] + 4)
                        if expand and co_pat.search(tk):
                            bx = _tight_company_box(gray, bx)
                        boxes.append(bx)
                continue
            box = (x0 - 4, y0 - 4, x1 + 4, y1 + 4)
            if expand and co_pat.search(joined):
                box = _tight_company_box(gray, box)
            boxes.append(box)
    return boxes


import re as _re_mod

_PADDLE_SENS = _re_mod.compile(
    r'公司|集團|集团|有限|股份|企業社|株式会社|实业|實業|'
    r'HSIEH|KUN|CO[\s.,]*LTD|CORP|INC|LIMITED|GMBH|ENGINEERING|'
    r'電話|傳真|电话|传真|TEL|FAX',
    _re_mod.IGNORECASE,
)


def paddle_redact_page(page) -> bool:
    """用 PaddleOCR(RapidOCR) 偵測中文/英文公司名、電話、客戶(TO)欄位並真實刪除。
    PaddleOCR 給的文字框已很精準，直接用該框做 redaction。回傳是否有遮蔽。"""
    ocr = get_rapidocr()
    if ocr is None:
        return False
    SCALE = 3.0
    pix = page.get_pixmap(matrix=fitz.Matrix(SCALE, SCALE), alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    try:
        res, _ = ocr(np.array(img))
    except Exception:
        return False
    if not res:
        return False
    co_pat = _re_mod.compile(r'公司|集團|集团|有限|股份|HSIEH|KUN|CO[\s.,]*LTD|CORP|INC|LIMITED|GMBH', _re_mod.IGNORECASE)
    allb = []
    to_box = None
    boxes = []
    co_boxes = []
    for box, text, score in res:
        xs = [pt[0] for pt in box]
        ys = [pt[1] for pt in box]
        bb = (min(xs), min(ys), max(xs), max(ys))
        allb.append((bb, text.strip()))
        if text.strip().upper() == "TO":
            to_box = bb
        if _PADDLE_SENS.search(text):
            boxes.append(bb)
        if co_pat.search(text):
            co_boxes.append(bb)
    # 商標緊貼公司名左側：往左延伸一塊涵蓋 LOGO（高度約等於公司名行高）
    for (x0, y0, x1, y1) in co_boxes:
        ch = y1 - y0
        lx1 = x0 - 2
        lx0 = max(0, x0 - int(ch * 2.2))
        if lx1 > lx0:
            boxes.append((lx0, y0 - int(ch * 0.3), lx1, y1 + int(ch * 0.3)))
    # 客戶欄位：TO 標籤右鄰格；TO 未讀到時取標題欄右上孤立英數短碼
    if to_box:
        c = [bb for bb, t in allb
             if bb[0] >= to_box[2] - 5
             and to_box[1] - 10 <= (bb[1] + bb[3]) / 2 <= to_box[3] + 10
             and bb[0] - to_box[2] < img.size[0] * 0.15]
        if c:
            boxes.append(min(c, key=lambda b: b[0]))
    else:
        H, W = img.size[1], img.size[0]
        tb = [(bb, t) for bb, t in allb
              if bb[1] > H * 0.80 and bb[0] > W * 0.80
              and _re_mod.fullmatch(r'[A-Za-z0-9]{2,8}', t)]
        if tb:
            boxes.append(min(tb, key=lambda x: x[0][1])[0])
    # 標題欄實心黑塊（LOGO，PaddleOCR 偵測不到圖形）
    boxes += dense_logo_boxes(img)
    if not boxes:
        return False
    inv = ~page.rotation_matrix
    for (x0, y0, x1, y1) in boxes:
        r = fitz.Rect(x0 / SCALE, y0 / SCALE, x1 / SCALE, y1 / SCALE) * inv
        r.normalize()
        r.x0 -= 2; r.y0 -= 2; r.x1 += 2; r.y1 += 2
        page.add_redact_annot(r, fill=(1, 1, 1))
    page.apply_redactions()
    return True


def redact_vector_page_ocr(page) -> bool:
    """向量頁找不到文字層公司名時：渲染→OCR→在敏感行位置做「真實 redaction」。
    用 add_redact_annot + apply_redactions 真正刪除底下文字/向量（非畫白框遮蓋），
    確保收件者無法選取/複製/還原被遮內容。回傳是否有遮蔽。"""
    SCALE = 3.0
    pix = page.get_pixmap(matrix=fitz.Matrix(SCALE, SCALE), alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    boxes = ocr_sensitive_boxes(img, expand=True)
    if boxes is None:
        return False
    # 追加：標題欄區域的實心黑塊（LOGO 密度遠高於文字/格線）
    boxes = list(boxes) + dense_logo_boxes(img)
    if not boxes:
        return False
    # pixmap 座標(已含旋轉) → 還原至頁面座標系
    inv = ~page.rotation_matrix
    for (x0, y0, x1, y1) in boxes:
        r = fitz.Rect(x0 / SCALE, y0 / SCALE, x1 / SCALE, y1 / SCALE) * inv
        r.normalize()
        page.add_redact_annot(r, fill=(1, 1, 1))
    page.apply_redactions()
    return True


def dense_logo_boxes(img):
    """在標題欄區域（底部 18%）找實心黑色區塊（LOGO）。
    以 24px 視窗密度 > 0.5 判定，相鄰視窗合併為框。"""
    gray = np.array(img.convert("L"))
    h, w = gray.shape
    y0 = int(h * 0.82)
    region = (gray[y0:h] < 100).astype(np.float32)
    WIN = 24
    rh, rw = region.shape
    if rh < WIN or rw < WIN:
        return []
    # 區塊平均（非重疊網格即可）
    gh, gw = rh // WIN, rw // WIN
    block = region[:gh * WIN, :gw * WIN].reshape(gh, WIN, gw, WIN).mean(axis=(1, 3))
    hits = np.argwhere(block > 0.4)
    if len(hits) == 0:
        return []
    # 合併相鄰網格
    boxes = []
    used = set()
    hitset = {(int(a), int(b)) for a, b in hits}
    for cell in hitset:
        if cell in used:
            continue
        stack = [cell]
        used.add(cell)
        comp = []
        while stack:
            cy, cx = stack.pop()
            comp.append((cy, cx))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nb = (cy + dy, cx + dx)
                    if nb in hitset and nb not in used:
                        used.add(nb)
                        stack.append(nb)
        ys = [c[0] for c in comp]; xs = [c[1] for c in comp]
        boxes.append((
            min(xs) * WIN - 6, y0 + min(ys) * WIN - 6,
            (max(xs) + 1) * WIN + 6, y0 + (max(ys) + 1) * WIN + 6,
        ))
    return boxes


def redact_scanned_page(doc, page_num: int, company_name: str, do_logo: bool, do_text: bool):
    """對掃描頁面做影像層面的遮蔽，再替換回 PDF。"""
    page = doc[page_num]
    # 優先取出內嵌原圖（保留原始解析度，避免重新渲染造成模糊）
    img = None
    imgs = page.get_images(full=True)
    if len(imgs) == 1:
        try:
            raw = doc.extract_image(imgs[0][0])
            img = Image.open(io.BytesIO(raw["image"])).convert("RGB")
        except Exception:
            img = None
    if img is None:
        mat = fitz.Matrix(3.0, 3.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    draw = ImageDraw.Draw(img)
    w, h = img.size
    dirty = False

    # ── 公司名稱：自動遮蔽 title block 或 OCR 偵測 ──────────────────────
    if do_text and not company_name.strip():
        # Auto mode: 先試 OCR 精準遮蔽敏感行（公司名/TEL/FAX/機密）；
        # OCR 不可用或無結果才退回標題欄頂線偵測（塗白整條）
        boxes = ocr_sensitive_boxes(img)
        if boxes:
            for (bx0, by0, bx1, by1) in boxes:
                draw.rectangle([bx0, by0, bx1, by1], fill="white")
            dirty = True
        else:
            gray = np.array(img.convert("L"))
            dark_frac = (gray < 128).mean(axis=1)
            tb_y0 = None
            y_lo, y_hi = int(h * 0.60), int(h * 0.97)
            for y in range(y_lo, y_hi):
                if dark_frac[y] > 0.55:
                    # 線以下還要有內容（文字列），排除最底部外框線
                    below = gray[y + 2:h]
                    if below.size and (below < 128).mean(axis=1).max() > 0.01 and (h - y) > h * 0.02:
                        tb_y0 = y
                        break
            if tb_y0 is None:
                tb_y0 = int(h * 0.80)
            draw.rectangle([0, tb_y0 + 1, w, h], fill="white")
            dirty = True

    if do_text and company_name.strip() and TESSERACT_OK:
        names = [n.strip() for n in company_name.split(",") if n.strip()]
        try:
            data = pytesseract.image_to_data(
                img, lang="chi_tra+chi_sim+eng",
                output_type=pytesseract.Output.DICT,
                config="--psm 11",
            )
            for name in names:
                # 嘗試在 OCR 輸出中找到完整或部分匹配
                full_text = " ".join(t for t in data["text"] if t.strip())
                # 逐個 token 查找
                for i, token in enumerate(data["text"]):
                    if not token.strip() or data["conf"][i] < 10:
                        continue
                    if token in name or name in token:
                        x, y = data["left"][i], data["top"][i]
                        bw, bh = data["width"][i], data["height"][i]
                        draw.rectangle([x - 3, y - 3, x + bw + 3, y + bh + 3], fill="white")
                        dirty = True
        except Exception:
            pass

    # ── LOGO：顏色偵測（在頁面四個角落的 title block 區）──────────────
    if do_logo:
        arr = np.array(img)
        CORNER = 0.20  # 頁面邊緣 20% 為 title block 候選區
        regions = [
            (0,            0,            int(w * CORNER),     int(h * CORNER)),
            (int(w * (1 - CORNER)), 0,  w,                   int(h * CORNER)),
            (0,            int(h * (1 - CORNER)), int(w * CORNER), h),
            (int(w * (1 - CORNER)), int(h * (1 - CORNER)), w, h),
        ]
        for rx0, ry0, rx1, ry1 in regions:
            region = arr[ry0:ry1, rx0:rx1]
            r_ch, g_ch, b_ch = region[:, :, 0], region[:, :, 1], region[:, :, 2]
            # 有彩度的像素（非黑白灰）
            sat = np.maximum(r_ch, np.maximum(g_ch, b_ch)).astype(int) - \
                  np.minimum(r_ch, np.minimum(g_ch, b_ch)).astype(int)
            # 也排除過暗（黑色）像素
            brightness = (r_ch.astype(int) + g_ch.astype(int) + b_ch.astype(int)) // 3
            colored = (sat > 35) & (brightness > 30)
            ys_c, xs_c = np.where(colored)
            if len(xs_c) < 30:
                continue
            # 找包圍框並白色遮蔽
            lx0, lx1 = int(xs_c.min()) + rx0, int(xs_c.max()) + rx0
            ly0, ly1 = int(ys_c.min()) + ry0, int(ys_c.max()) + ry0
            draw.rectangle([lx0 - 4, ly0 - 4, lx1 + 4, ly1 + 4], fill="white")
            dirty = True

    if not dirty:
        return

    # ── 把修改後的影像寫回 PDF 頁面 ────────────────────────────────────
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    # 建立暫時 doc 取代該頁
    tmp = fitz.open()
    tmp_page = tmp.new_page(width=page.rect.width, height=page.rect.height)
    tmp_page.insert_image(tmp_page.rect, stream=buf.read(), keep_proportion=False)

    doc.delete_page(page_num)
    doc.insert_pdf(tmp, from_page=0, to_page=0, start_at=page_num)
    tmp.close()


def process_doc(doc, service: str, rotate_deg: int, company_name: str):
    do_rotate = service in ("rotate", "full")
    do_redact = service in ("redact", "full")

    if do_rotate:
        for page in doc:
            page.set_rotation((page.rotation + rotate_deg) % 360)

    if do_redact:
        remove_watermark_artifacts(doc)
        # 先處理向量頁面的文字和 LOGO
        text_was_redacted = apply_text_redaction(doc, company_name)
        apply_logo_redaction(doc)

        for i in range(len(doc)):
            page_is_scanned = is_scanned_page(doc[i])
            if page_is_scanned:
                # 掃描頁：影像層遮蔽
                redact_scanned_page(doc, i,
                    company_name=company_name,
                    do_logo=True,
                    do_text=True
                )
            elif not company_name.strip():
                # Auto mode：向量頁跑 OCR 補強。優先 PaddleOCR（中文辨識佳）；
                # 成功就不重跑 Tesseract（省一半時間）；PaddleOCR 無結果才退回。
                paddle_hit = paddle_redact_page(doc[i])
                ocr_hit = paddle_hit or redact_vector_page_ocr(doc[i])
                # OCR 全不可用且文字層也沒抓到 → 退回影像層 title block 整條遮蔽
                if (not ocr_hit and not text_was_redacted
                        and not TESSERACT_OK and not RAPIDOCR_OK):
                    redact_scanned_page(doc, i,
                        company_name="",
                        do_logo=False,
                        do_text=True
                    )


# ── Upload (pre-payment file staging) ───────────────────────────────────────

ALLOWED_EXTS = {".pdf", ".jpg", ".jpeg", ".png"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, "只接受 PDF / JPG / PNG 檔案")

    upload_id = str(uuid.uuid4())
    stage_path = UPLOAD_DIR / f"{upload_id}_stage.pdf"
    content = await file.read()

    if ext in (".jpg", ".jpeg", ".png"):
        # 圖片包裝成 PDF（以 150 DPI 計算頁面大小）
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
        iw, ih = pil_img.size
        pdf_w = iw * 72 / 150
        pdf_h = ih * 72 / 150
        tmp_doc = fitz.open()
        tmp_page = tmp_doc.new_page(width=pdf_w, height=pdf_h)
        tmp_page.insert_image(tmp_page.rect, stream=content, keep_proportion=False)
        tmp_doc.save(str(stage_path))
        tmp_doc.close()
    else:
        with open(stage_path, "wb") as f:
            f.write(content)

    asyncio.create_task(auto_delete(str(stage_path), 1800))
    return {"upload_id": upload_id, "filename": file.filename}


# ── Stripe Checkout ──────────────────────────────────────────────────────────

@app.post("/create-checkout")
async def create_checkout(request: Request):
    if not stripe.api_key:
        raise HTTPException(503, "Stripe 尚未設定")

    body = await request.json()
    upload_ids = body.get("upload_ids", [])
    service = body.get("service", "rotate")
    rotate_deg = int(body.get("rotate_deg", 90))
    company_name = body.get("company_name", "")

    if not upload_ids:
        raise HTTPException(400, "未提供檔案")

    unit_cents = PRICE_CENTS.get(service, 300)
    total_cents = unit_cents * len(upload_ids)

    session = stripe.checkout.Session.create(
        payment_method_types=["card"],
        line_items=[{
            "price_data": {
                "currency": "usd",
                "product_data": {
                    "name": f"DrawShield — {SERVICE_NAMES.get(service, service)} × {len(upload_ids)} 份"
                },
                "unit_amount": total_cents,
            },
            "quantity": 1,
        }],
        mode="payment",
        success_url=(
            f"{FRONTEND_URL}/?paid=1"
            f"&session_id={{CHECKOUT_SESSION_ID}}"
            f"&service={service}"
            f"&rotate_deg={rotate_deg}"
            f"&upload_ids={'|'.join(upload_ids)}"
            f"&cname={company_name}"
        ),
        cancel_url=f"{FRONTEND_URL}/",
    )

    return {"checkout_url": session.url}


# ── 綠界 ECPay 工具函式 ───────────────────────────────────────────────────

def _ecpay_check_mac(params: dict) -> str:
    """依綠界 AIO V5 規格計算 CheckMacValue（SHA256, 大寫）。"""
    import hashlib
    from urllib.parse import quote_plus
    items = sorted((k, v) for k, v in params.items() if k != "CheckMacValue")
    raw = f"HashKey={ECPAY_HASH_KEY}&" + \
          "&".join(f"{k}={v}" for k, v in items) + \
          f"&HashIV={ECPAY_HASH_IV}"
    encoded = quote_plus(raw).lower()
    # .NET UrlEncode 相容字元還原
    for a, b in (("%2d", "-"), ("%5f", "_"), ("%2e", "."), ("%21", "!"),
                 ("%2a", "*"), ("%28", "("), ("%29", ")")):
        encoded = encoded.replace(a, b)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest().upper()


async def _process_uploads(upload_ids, service, rotate_deg, company_name):
    """共用處理流程：對暫存檔做去識別，回傳 [{upload_id, download_id}]。"""
    results = []
    for upload_id in upload_ids:
        try:
            uuid.UUID(upload_id)
        except ValueError:
            raise HTTPException(400, f"無效的 upload_id：{upload_id}")
        stage_path = UPLOAD_DIR / f"{upload_id}_stage.pdf"
        if not stage_path.exists():
            raise HTTPException(404, f"暫存檔案已過期，請重新上傳：{upload_id}")
        job_id = str(uuid.uuid4())
        out_path = UPLOAD_DIR / f"{job_id}_out.pdf"
        try:
            doc = fitz.open(str(stage_path))
            process_doc(doc, service, rotate_deg, company_name)
            doc.save(str(out_path))
            doc.close()
        except Exception as e:
            raise HTTPException(500, f"處理失敗：{str(e)}")
        finally:
            asyncio.create_task(auto_delete(str(stage_path), 5))
        asyncio.create_task(auto_delete(str(out_path), 600))
        results.append({"upload_id": upload_id, "download_id": job_id})
    return results


# ── 綠界 ECPay 結帳 ───────────────────────────────────────────────────────

@app.post("/create-ecpay-order")
async def create_ecpay_order(request: Request):
    """建立綠界訂單，回傳自動送出表單所需的 action 與參數。"""
    import time
    body = await request.json()
    upload_ids = body.get("upload_ids", [])
    service = body.get("service", "rotate")
    rotate_deg = int(body.get("rotate_deg", 90))
    company_name = body.get("company_name", "")
    if not upload_ids:
        raise HTTPException(400, "未提供檔案")

    unit = PRICE_TWD.get(service, 100)
    total = unit * len(upload_ids)
    # MerchantTradeNo：英數，≤20 碼
    trade_no = "DS" + uuid.uuid4().hex[:16].upper()
    ECPAY_ORDERS[trade_no] = {
        "upload_ids": upload_ids, "service": service,
        "rotate_deg": rotate_deg, "company_name": company_name,
        "paid": False, "results": None,
    }
    asyncio.create_task(_expire_order(trade_no, 1800))

    svc = SERVICE_NAMES.get(service, service)
    params = {
        "MerchantID": ECPAY_MERCHANT_ID,
        "MerchantTradeNo": trade_no,
        "MerchantTradeDate": time.strftime("%Y/%m/%d %H:%M:%S"),
        "PaymentType": "aio",
        "TotalAmount": str(total),
        "TradeDesc": "DrawShield 圖面去識別服務",
        "ItemName": f"DrawShield {svc} x{len(upload_ids)}",
        "ReturnURL": f"{API_PUBLIC_URL}/ecpay-return",
        "ClientBackURL": f"{FRONTEND_URL}/?ecpay_order={trade_no}",
        "ChoosePayment": "Credit",
        "EncryptType": "1",
    }
    params["CheckMacValue"] = _ecpay_check_mac(params)
    return {"action": ECPAY_AIO_URL, "params": params, "order": trade_no}


@app.post("/ecpay-return")
async def ecpay_return(request: Request):
    """綠界 server-to-server 付款結果回呼。驗章後標記訂單已付款。"""
    form = await request.form()
    data = {k: v for k, v in form.items()}
    mac = data.get("CheckMacValue", "")
    if _ecpay_check_mac(data) != mac:
        return PlainTextResponse("0|CheckMacValue Error")
    trade_no = data.get("MerchantTradeNo", "")
    order = ECPAY_ORDERS.get(trade_no)
    if order is None:
        return PlainTextResponse("0|Order Not Found")
    if data.get("RtnCode") == "1":  # 付款成功
        if not order["paid"]:
            order["paid"] = True
            try:
                order["results"] = await _process_uploads(
                    order["upload_ids"], order["service"],
                    order["rotate_deg"], order["company_name"],
                )
            except Exception:
                order["results"] = []
    return PlainTextResponse("1|OK")


@app.get("/ecpay-order-status/{trade_no}")
async def ecpay_order_status(trade_no: str):
    order = ECPAY_ORDERS.get(trade_no)
    if order is None:
        raise HTTPException(404, "訂單不存在或已過期")
    return {"paid": order["paid"], "results": order["results"] or []}


async def _expire_order(trade_no: str, delay: int):
    await asyncio.sleep(delay)
    ECPAY_ORDERS.pop(trade_no, None)


# ── Process after payment ────────────────────────────────────────────────────

@app.post("/process-paid")
async def process_paid(request: Request):
    body = await request.json()
    session_id = body.get("session_id", "")
    upload_ids = body.get("upload_ids", [])
    service = body.get("service", "rotate")
    rotate_deg = int(body.get("rotate_deg", 90))
    company_name = body.get("company_name", "")

    if not upload_ids:
        raise HTTPException(400, "未提供檔案 ID")

    # Verify payment
    if stripe.api_key:
        if not session_id:
            raise HTTPException(402, "需要付款驗證")
        try:
            session = stripe.checkout.Session.retrieve(session_id)
            if session.payment_status != "paid":
                raise HTTPException(402, "付款未完成")
        except stripe.error.StripeError as e:
            raise HTTPException(402, f"付款驗證失敗：{str(e)}")

    results = []
    for upload_id in upload_ids:
        try:
            uuid.UUID(upload_id)
        except ValueError:
            raise HTTPException(400, f"無效的 upload_id：{upload_id}")

        stage_path = UPLOAD_DIR / f"{upload_id}_stage.pdf"
        if not stage_path.exists():
            raise HTTPException(404, f"暫存檔案已過期，請重新上傳：{upload_id}")

        job_id = str(uuid.uuid4())
        out_path = UPLOAD_DIR / f"{job_id}_out.pdf"

        try:
            doc = fitz.open(str(stage_path))
            process_doc(doc, service, rotate_deg, company_name)
            doc.save(str(out_path))
            doc.close()
        except Exception as e:
            raise HTTPException(500, f"處理失敗：{str(e)}")
        finally:
            asyncio.create_task(auto_delete(str(stage_path), 5))

        asyncio.create_task(auto_delete(str(out_path), 600))
        results.append({"upload_id": upload_id, "download_id": job_id})

    return {"results": results}


# ── Direct process (no Stripe, for testing) ──────────────────────────────────

@app.post("/process")
async def process_pdf(
    file: UploadFile = File(...),
    service: str = Form(...),
    rotate_deg: int = Form(90),
    company_name: str = Form(""),
):
    if stripe.api_key:
        raise HTTPException(402, "請透過付款流程處理")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "只接受 PDF 檔案")

    job_id = str(uuid.uuid4())
    in_path = UPLOAD_DIR / f"{job_id}_in.pdf"
    out_path = UPLOAD_DIR / f"{job_id}_out.pdf"

    content = await file.read()
    with open(in_path, "wb") as f:
        f.write(content)

    try:
        doc = fitz.open(str(in_path))
        process_doc(doc, service, rotate_deg, company_name)
        doc.save(str(out_path))
        doc.close()
    except Exception as e:
        raise HTTPException(500, f"處理失敗：{str(e)}")
    finally:
        asyncio.create_task(auto_delete(str(in_path), 60))

    asyncio.create_task(auto_delete(str(out_path), 600))
    return {"download_id": job_id}


# ── Download ─────────────────────────────────────────────────────────────────

@app.get("/download/{job_id}")
async def download(job_id: str):
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(400, "無效的 ID")

    path = UPLOAD_DIR / f"{job_id}_out.pdf"
    if not path.exists():
        raise HTTPException(404, "檔案不存在或已過期")

    return FileResponse(str(path), media_type="application/pdf", filename="processed.pdf")


@app.post("/debug-spans")
async def debug_spans(file: UploadFile = File(...), company_name: str = Form("")):
    """Debug: return title block spans and search_for results."""
    import re as _re
    data = await file.read()
    doc = fitz.open(stream=data, filetype="pdf")
    page = doc[0]
    w, h = page.rect.width, page.rect.height
    spans = []
    for b in page.get_text("dict")["blocks"]:
        if b["type"] == 0:
            for line in b["lines"]:
                for span in line["spans"]:
                    x0,y0,x1,y1 = span["bbox"]
                    if x0 < w*0.25 and y0 > 550:
                        txt = span["text"]
                        clean = ''.join(c for c in txt if ord(c) < 0xD800 or ord(c) > 0xDFFF)
                        spans.append({
                            "text": txt,
                            "clean": clean,
                            "codepoints": [hex(ord(c)) for c in txt[:8]],
                            "bbox": [round(v,1) for v in span["bbox"]],
                        })
    search_results = {}
    if company_name:
        for name in company_name.split(","):
            name = name.strip()
            if name:
                rects = page.search_for(name)
                search_results[name] = [[round(v,1) for v in r] for r in rects]
    return {"spans": spans, "search_results": search_results}


@app.get("/")
def root():
    return {"status": "ok", "service": "DrawShield API", "version": VERSION}
