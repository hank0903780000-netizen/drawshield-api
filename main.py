from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import fitz  # pymupdf
import uuid, os, asyncio, shutil
from pathlib import Path

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("/tmp/drawshield")
UPLOAD_DIR.mkdir(exist_ok=True)

async def auto_delete(path: str, delay: int = 60):
    await asyncio.sleep(delay)
    try:
        os.remove(path)
    except:
        pass

@app.get("/")
def root():
    return {"status": "DrawShield API running"}

@app.post("/process")
async def process_pdf(
    file: UploadFile = File(...),
    service: str = Form(...),        # rotate | redact | both
    rotate_deg: int = Form(90),      # 90 | 180 | 270
    company_name: str = Form(""),    # 要遮蔽的名稱，逗號分隔
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "只接受 PDF 檔案")

    job_id = str(uuid.uuid4())
    in_path  = UPLOAD_DIR / f"{job_id}_in.pdf"
    out_path = UPLOAD_DIR / f"{job_id}_out.pdf"

    # 儲存上傳檔案
    content = await file.read()
    with open(in_path, "wb") as f:
        f.write(content)

    try:
        doc = fitz.open(str(in_path))

        # ── 旋轉 ──────────────────────────────────────────
        if service in ("rotate", "both"):
            for page in doc:
                current = page.rotation
                page.set_rotation((current + rotate_deg) % 360)

        # ── 遮蔽公司名稱 ──────────────────────────────────
        if service in ("redact", "both") and company_name.strip():
            names = [n.strip() for n in company_name.split(",") if n.strip()]
            for page in doc:
                for name in names:
                    # 逐字搜尋（相容直書圖框）
                    areas = []
                    blocks = page.get_text("dict")["blocks"]
                    for b in blocks:
                        if b["type"] != 0:
                            continue
                        for line in b["lines"]:
                            for span in line["spans"]:
                                if any(c in span["text"] for c in list(name)):
                                    x0,y0,x1,y1 = span["bbox"]
                                    # 只蓋在合理 x 範圍內（避免誤蓋「公差」等字）
                                    # 用 span text 精確比對
                                    if span["text"].strip() in list(name):
                                        areas.append(fitz.Rect(x0-3,y0-3,x1+3,y1+3))
                    for r in areas:
                        page.draw_rect(r, color=(1,1,1), fill=(1,1,1))

        doc.save(str(out_path))
        doc.close()

    except Exception as e:
        raise HTTPException(500, f"處理失敗：{str(e)}")
    finally:
        # 60秒後刪除輸入檔
        asyncio.create_task(auto_delete(str(in_path), 60))

    # 輸出檔 10 分鐘後刪除
    asyncio.create_task(auto_delete(str(out_path), 600))

    return {"download_id": job_id}


@app.get("/download/{job_id}")
async def download(job_id: str):
    # 簡單驗證 job_id 格式
    try:
        uuid.UUID(job_id)
    except:
        raise HTTPException(400, "無效的 ID")

    path = UPLOAD_DIR / f"{job_id}_out.pdf"
    if not path.exists():
        raise HTTPException(404, "檔案不存在或已過期")

    return FileResponse(
        str(path),
        media_type="application/pdf",
        filename="processed.pdf"
    )
