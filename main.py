from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
import fitz
import stripe
import uuid, os, asyncio
from pathlib import Path

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


async def auto_delete(path: str, delay: int = 60):
    await asyncio.sleep(delay)
    try:
        os.remove(path)
    except Exception:
        pass


def apply_text_redaction(doc, company_name: str):
    names = [n.strip() for n in company_name.split(",") if n.strip()]
    for page in doc:
        for name in names:
            rects = page.search_for(name)
            for rect in rects:
                page.add_redact_annot(rect, fill=(1, 1, 1))
        page.apply_redactions()


def apply_logo_redaction(doc):
    for page in doc:
        page_area = page.rect.width * page.rect.height
        img_list = page.get_images(full=True)
        for img in img_list:
            xref = img[0]
            try:
                rects = page.get_image_rects(xref)
            except Exception:
                continue
            for rect in rects:
                img_area = rect.width * rect.height
                # Skip full-page images (scanned PDFs where whole page is one image)
                if page_area > 0 and img_area / page_area > 0.5:
                    continue
                page.add_redact_annot(rect, fill=(1, 1, 1))
        page.apply_redactions()


def process_doc(doc, service: str, rotate_deg: int, company_name: str):
    if service in ("rotate", "full"):
        for page in doc:
            page.set_rotation((page.rotation + rotate_deg) % 360)

    if service in ("redact", "full") and company_name.strip():
        apply_text_redaction(doc, company_name)

    if service in ("redact", "full"):
        apply_logo_redaction(doc)


# ── Upload (pre-payment file staging) ───────────────────────────────────────

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "只接受 PDF 檔案")

    upload_id = str(uuid.uuid4())
    stage_path = UPLOAD_DIR / f"{upload_id}_stage.pdf"

    content = await file.read()
    with open(stage_path, "wb") as f:
        f.write(content)

    # Auto-delete staged file after 30 minutes (enough time for checkout)
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


@app.get("/")
def root():
    return {"status": "ok", "service": "DrawShield API"}
