from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
import fitz
import uuid, os, asyncio
from pathlib import Path
 
app = FastAPI()
 
# 允許所有 host
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)
 
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
    except:
        pass
 
@app.get("/")
def root():
    return {"status": "ok", "service": "DrawShield API"}
 
@app.post("/process")
async def process_pdf(
    file: UploadFile = File(...),
    service: str = Form(...),
    rotate_deg: int = Form(90),
    company_name: str = Form(""),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "只接受 PDF 檔案")
 
    job_id = str(uuid.uuid4())
    in_path  = UPLOAD_DIR / f"{job_id}_in.pdf"
    out_path = UPLOAD_DIR / f"{job_id}_out.pdf"
 
    content = await file.read()
    with open(in_path, "wb") as f:
        f.write(content)
 
    try:
        doc = fitz.open(str(in_path))
 
        if service in ("rotate", "both"):
            for page in doc:
                current = page.rotation
                page.set_rotation((current + rotate_deg) % 360)
 
        if service in ("redact", "both") and company_name.strip():
            names = [n.strip() for n in company_name.split(",") if n.strip()]
            for page in doc:
                for name in names:
                    chars = list(name)
                    blocks = page.get_text("dict")["blocks"]
                    for b in blocks:
                        if b["type"] != 0:
                            continue
                        for line in b["lines"]:
                            for span in line["spans"]:
                                if span["text"].strip() in chars:
                                    x0,y0,x1,y1 = span["bbox"]
                                    r = fitz.Rect(x0-3, y0-3, x1+3, y1+3)
                                    page.draw_rect(r, color=(1,1,1), fill=(1,1,1))
 
        doc.save(str(out_path))
        doc.close()
 
    except Exception as e:
        raise HTTPException(500, f"處理失敗：{str(e)}")
    finally:
        asyncio.create_task(auto_delete(str(in_path), 60))
 
    asyncio.create_task(auto_delete(str(out_path), 600))
    return {"download_id": job_id}
 
 
@app.get("/download/{job_id}")
async def download(job_id: str):
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
        filename="processed.pdf",
    )
 
