"""FastAPI app: startup checks and routes. Run with: uvicorn app.main:app"""
import logging
import shutil
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import Depends, FastAPI, File, Header, Query, Request, UploadFile
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

from .config import get_settings
from .errors import AppError
from .label_store import LabelStore, open_collection
from .llm import OllamaClient
from .logging_setup import request_id_var, setup_logging
from .ocr import OcrEngine
from .pipeline import Pipeline
from .prompts import PROMPTS
from .schemas import ConfirmRequest, RejectRequest
from .storage import Storage

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = get_settings()
    settings.ensure_dirs()
    if settings.prompt_version not in PROMPTS:
        raise RuntimeError(f"Unknown PROMPT_VERSION '{settings.prompt_version}'")
    if settings.api_key == "change-me":
        log.warning("API_KEY is still the default; set a real one in .env")

    db = Storage(settings.db_path)
    llm = OllamaClient(settings)
    try:
        missing = [m for m in (settings.llm_model, settings.embed_model) if not llm.has_model(m)]
    except AppError as e:
        raise RuntimeError(f"{e.message}. Start Ollama, then start this server again.") from e
    if missing:
        raise RuntimeError("Missing Ollama models. Run: " + " && ".join(f"ollama pull {m}" for m in missing))

    log.info("warming up models (the first load on CPU can take a minute)")
    llm.embed(["warm up"])
    llm.generate("Say OK.", json_mode=False, num_predict=3)

    collection = open_collection(settings)  # raises if EMBED_MODEL changed without a reindex
    store = LabelStore(db, settings, embed=llm.embed, collection=collection)
    if len(store) == 0:
        log.warning("the label library is empty; run: python scripts/load_seed.py")
    log.info("label library: %d labels, %d vectors", len(store), collection.count())

    app.state.settings = settings
    app.state.db = db
    app.state.llm = llm
    app.state.store = store
    app.state.ocr = OcrEngine(settings)
    app.state.pipeline = Pipeline(settings, db, llm, store, app.state.ocr)
    yield


app = FastAPI(title="Invoice Label-RAG", lifespan=lifespan)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        description=app.description,
        routes=app.routes,
    )
    for schema in openapi_schema.get("components", {}).get("schemas", {}).values():
        if not isinstance(schema, dict):
            continue
        props = schema.get("properties", {})
        for prop_name, prop_val in props.items():
            if isinstance(prop_val, dict):
                if prop_val.get("contentMediaType") == "application/octet-stream":
                    prop_val["format"] = "binary"
                items = prop_val.get("items")
                if isinstance(items, dict) and items.get("contentMediaType") == "application/octet-stream":
                    items["format"] = "binary"
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


def _error_body(message: str, **extra) -> dict:
    return {"request_id": request_id_var.get(), "status": "error", "error": message, **extra}


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id_var.set(uuid.uuid4().hex)
    try:
        response = await call_next(request)
    except Exception:
        log.exception("unexpected error")
        response = JSONResponse(_error_body("Unexpected server error"), status_code=500)
    response.headers["X-Request-ID"] = request_id_var.get()
    return response


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    log.warning("request failed (%d): %s", exc.status_code, exc.message)
    return JSONResponse(_error_body(exc.message, **exc.extra), status_code=exc.status_code)


def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    if x_api_key != request.app.state.settings.api_key:
        raise AppError(401, "Missing or wrong API key (send it in the X-API-Key header)")


def get_pipeline(request: Request) -> Pipeline:
    return request.app.state.pipeline


auth = [Depends(require_api_key)]


# Route functions are plain `def`, so FastAPI runs them in worker threads and the
# event loop stays responsive while OCR and the LLM are busy.

@app.get("/health")
def health(request: Request):
    st = request.app.state
    checks = {}
    try:
        models = st.llm.list_models()
        checks["ollama"] = {
            "ok": True,
            "llm_model": st.llm.has_model(st.settings.llm_model),
            "embed_model": st.llm.has_model(st.settings.embed_model),
            "models": models,
        }
    except AppError as e:
        checks["ollama"] = {"ok": False, "error": e.message}
    try:
        checks["chromadb"] = {"ok": True, "vectors": st.store.collection.count()}
    except Exception as e:
        checks["chromadb"] = {"ok": False, "error": str(e)}
    try:
        checks["sqlite"] = {"ok": True, "labels": st.db.one("SELECT COUNT(*) AS n FROM labels")["n"]}
    except Exception as e:
        checks["sqlite"] = {"ok": False, "error": str(e)}
    checks["easyocr"] = {"ok": True, "loaded": st.ocr.loaded}
    free_gb = shutil.disk_usage(st.settings.data_dir).free / 1e9
    checks["disk"] = {"ok": free_gb > 1, "free_gb": round(free_gb, 1)}
    healthy = all(c["ok"] for c in checks.values())
    return JSONResponse({"status": "ok" if healthy else "degraded", "checks": checks},
                        status_code=200 if healthy else 503)


@app.post("/ingest", dependencies=auth)
async def ingest(files: list[UploadFile] = File(..., description="Invoice PDF or image files"), pipeline: Pipeline = Depends(get_pipeline)):
    results = []
    for f in files:
        try:
            contents = await f.read()
            results.append(pipeline.ingest(contents, f.filename or "upload"))
        except AppError as e:  # one bad file must not stop the batch
            log.warning("ingest of %s failed (%d): %s", f.filename, e.status_code, e.message)
            results.append({**_error_body(e.message, **e.extra), "filename": f.filename,
                            "http_status": e.status_code})
        except Exception:
            log.exception("ingest of %s failed", f.filename)
            results.append({**_error_body("Unexpected server error"), "filename": f.filename, "http_status": 500})
    return {"request_id": request_id_var.get(), "results": results}


@app.get("/pending", dependencies=auth)
def list_pending(request: Request, status: str = "pending"):
    return {"records": request.app.state.db.list_pending(status)}


@app.get("/pending/{doc_id}", dependencies=auth)
def get_pending(doc_id: str, request: Request):
    rec = request.app.state.db.get_pending(doc_id)
    if rec is None:
        raise AppError(404, f"No pending record with doc_id {doc_id}")
    return rec


@app.post("/confirm", dependencies=auth)
def confirm(req: ConfirmRequest, pipeline: Pipeline = Depends(get_pipeline)):
    return pipeline.confirm(req)


@app.post("/reject", dependencies=auth)
def reject(req: RejectRequest, request: Request):
    db = request.app.state.db
    rec = db.get_pending(req.doc_id)
    if rec is None:
        raise AppError(404, f"No pending record with doc_id {req.doc_id}")
    if rec["status"] != "pending":
        raise AppError(409, f"Record {req.doc_id} is already {rec['status']}")
    db.set_pending_status(req.doc_id, "rejected")
    return {"doc_id": req.doc_id, "status": "rejected"}


@app.get("/labels", dependencies=auth)
def list_labels(request: Request, field: str | None = None, source: str | None = None, min_count: int = 0):
    labels = [
        asdict(r) for r in request.app.state.store.all()
        if (field is None or r.field == field)
        and (source is None or r.source == source)
        and r.count >= min_count
    ]
    labels.sort(key=lambda r: (r["field"], -r["count"]))
    return {"count": len(labels), "labels": labels}


@app.delete("/labels/{label_id}", dependencies=auth)
def delete_label(label_id: int, request: Request):
    if not request.app.state.store.delete(label_id):
        raise AppError(404, f"No label with id {label_id}")
    return {"deleted": label_id}


@app.get("/vendors", dependencies=auth)
def list_vendors(request: Request):
    rows = request.app.state.db.query("SELECT * FROM vendors ORDER BY name")
    for row in rows:
        row.pop("bank_hash", None)  # only the last 4 digits are shown
    return {"vendors": rows}


@app.post("/extract", dependencies=auth)
async def extract(
    file: UploadFile = File(..., description="Invoice PDF or image file"),
    hints: bool | None = Query(default=None, description="Override HINTS_ENABLED for this request"),
    pipeline: Pipeline = Depends(get_pipeline),
):
    enabled = pipeline.settings.hints_enabled if hints is None else hints
    contents = await file.read()
    return pipeline.extract(contents, file.filename or "upload", "all" if enabled else "off")
