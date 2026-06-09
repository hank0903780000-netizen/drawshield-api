from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
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

app = FastAPI()

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "https://drawshield.vercel.app")

PRICE_CENTS = {"rotate": 300, "redact": 300, "full": 500}
SERVICE_NAMES = {
    "rotate": "PDF 旋轉",
    "redact": "遮蔽公司名稱和 LOGO",
    "full": "旋轉 + 遮蔽公司名稱和 LOGO",
}

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
VERSION = "ffc4016-surrogate-fix"


async def auto_delete(path: str, delay: int = 60):
    await asyncio.sleep(delay)
    try:
        os.remove(path)
    except Exception:
        pass


def apply_text_redaction(doc, company_name: str):
    import re
    names = [n.strip() for n in company_name.split(",") if n.strip()]
    phone_pattern = re.compile(r"(?:TEL|FAX|AX|電話|傳真)\s*[:：]?\s*[\d\-\+\(\) ]{5,}", re.IGNORECASE)

    for page in doc:
        w, h = page.rect.width, page.rect.height
        # 收集所有 title block 區域的 span（頁面左側 25% 或下方 25%）
        all_spans = []
        for b in page.get_text("dict")["blocks"]:
            if b["type"] != 0:
                continue
            for line in b["lines"]:
                for span in line["spans"]:
                    x0, y0, x1, y1 = span["bbox"]
                    if x0 < w * 0.25 or y0 > h * 0.75:
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

        # 找到匹配的 span，並擴展遮蔽同一欄位（相同 x 範圍）的所有文字
        matched_x_bands = []  # [(x0, x1, y0, y1)] 已匹配的欄位範圍
        for name in names:
            # 方法一：search_for（英文/簡單文字效果好）
            rects = page.search_for(name)
            for rect in rects:
                # 找到 search_for 命中的 span，遮蔽整個 span（不只是部分 rect）
                span_found = False
                for span in all_spans:
                    if rect_contains_span(rect, span["bbox"]):
                        full_rect = fitz.Rect(span["bbox"])
                        page.add_redact_annot(full_rect, fill=(1, 1, 1))
                        matched_x_bands.append((full_rect.x0 - 5, full_rect.x1 + 5, full_rect.y0, full_rect.y1))
                        span_found = True
                if not span_found:
                    page.add_redact_annot(rect, fill=(1, 1, 1))
                    matched_x_bands.append((rect.x0 - 5, rect.x1 + 5, rect.y0, rect.y1))
            # 方法二：span 文字比對（處理 search_for 完全找不到的情況）
            if not rects:
                for span in all_spans:
                    if text_matches(span["text"], name):
                        full_rect = fitz.Rect(span["bbox"])
                        page.add_redact_annot(full_rect, fill=(1, 1, 1))
                        matched_x_bands.append((full_rect.x0 - 5, full_rect.x1 + 5, full_rect.y0, full_rect.y1))

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

        page.apply_redactions()


def apply_logo_redaction(doc):
    for page in doc:
        page_area = page.rect.width * page.rect.height
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
        pw = page.rect.width
        ph = page.rect.height
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


def redact_scanned_page(doc, page_num: int, company_name: str, do_logo: bool, do_text: bool):
    """對掃描頁面做影像層面的遮蔽，再替換回 PDF。"""
    page = doc[page_num]
    SCALE = 2.0
    mat = fitz.Matrix(SCALE, SCALE)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    draw = ImageDraw.Draw(img)
    w, h = img.size
    dirty = False

    # ── 公司名稱：OCR 偵測 ──────────────────────────────────────────────
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
        # 先處理向量頁面的文字和 LOGO
        apply_text_redaction(doc, company_name)
        apply_logo_redaction(doc)

        # 再處理掃描頁面（影像層）
        for i in range(len(doc)):
            if is_scanned_page(doc[i]):
                redact_scanned_page(doc, i,
                    company_name=company_name,
                    do_logo=True,
                    do_text=bool(company_name.strip())
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
