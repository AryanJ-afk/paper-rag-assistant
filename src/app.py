from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import time
import uuid
import logging
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from rag import ensure_dirs, PDF_DIR, ingest_pdf, ask, summarize_paper

app = FastAPI(title="Research Paper RAG Assistant (Local)")

BASE_DIR = Path(__file__).resolve().parent.parent  # <-- project root
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("rag")

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    source_file: str | None = None

@app.middleware("http")
async def log_requests(request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.time()

    response = await call_next(request)

    dt_ms = (time.time() - start) * 1000
    logger.info(
        f'{{"request_id":"{request_id}","method":"{request.method}","path":"{request.url.path}",'
        f'"status":{response.status_code},"latency_ms":{dt_ms:.1f}}}'
    )
    response.headers["X-Request-ID"] = request_id
    return response

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    ensure_dirs()
    dst = PDF_DIR / file.filename
    content = await file.read()
    dst.write_bytes(content)
    result = ingest_pdf(file.filename)
    return result

@app.post("/query")
def query(req: QueryRequest):
    return ask(req.question, top_k=req.top_k, source_file=req.source_file)

class SummarizeRequest(BaseModel):
    source_file: str

@app.post("/summarize")
def summarize(req: SummarizeRequest):
    return summarize_paper(req.source_file)