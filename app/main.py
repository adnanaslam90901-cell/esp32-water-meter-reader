from __future__ import annotations

import re
import shutil
import sqlite3
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from threading import Event

import cv2
from fastapi import Body, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.datastructures import UploadFile as StarletteUploadFile

from . import dataset as ds
from .reader import MeterReader

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CAPTURES = DATA / "captures"
BATCH_UPLOADS = DATA / "batch_uploads"
MODELS = ROOT / "models"
DB = DATA / "readings.db"
CAPTURES.mkdir(parents=True, exist_ok=True)
reader = MeterReader(DATA / "config.json")
app = FastAPI(title="ESP32 Water Meter Reader", version="1.1.0")
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")
templates = Jinja2Templates(directory=ROOT / "app" / "templates")
batch_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="batch")
cancel_flags: dict[int, Event] = {}


def connect():
    db = sqlite3.connect(DB, timeout=15)
    db.row_factory = sqlite3.Row
    return db


@app.on_event("startup")
def startup():
    with closing(connect()) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT, meter_id TEXT NOT NULL,
            reading TEXT NOT NULL, confidence REAL NOT NULL, image_name TEXT NOT NULL,
            roi_source TEXT NOT NULL, created_at INTEGER NOT NULL, corrected INTEGER NOT NULL DEFAULT 0
        )""")
        existing = {row["name"] for row in db.execute("PRAGMA table_info(readings)")}
        for column, kind in (("batch_id", "INTEGER"), ("source_name", "TEXT"), ("expected", "TEXT"), ("note", "TEXT")):
            if column not in existing:
                db.execute(f"ALTER TABLE readings ADD COLUMN {column} {kind}")
        db.execute("CREATE INDEX IF NOT EXISTS readings_batch ON readings(batch_id)")
        db.execute("""CREATE TABLE IF NOT EXISTS batches (
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, meter_id TEXT NOT NULL,
            status TEXT NOT NULL, total INTEGER NOT NULL DEFAULT 0, processed INTEGER NOT NULL DEFAULT 0,
            failed INTEGER NOT NULL DEFAULT 0, labelled INTEGER NOT NULL DEFAULT 0,
            created_at INTEGER NOT NULL, finished_at INTEGER
        )""")
        # A batch cannot survive a restart: its worker thread is gone.
        db.execute("UPDATE batches SET status='interrupted', finished_at=? WHERE status IN ('queued','running')", (int(time.time()),))
        db.commit()
    shutil.rmtree(BATCH_UPLOADS, ignore_errors=True)


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})


def process_image(data: bytes, meter_id: str):
    if len(data) > ds.MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image is larger than 12 MB")
    try:
        image = reader.decode(data)
        result = reader.read(image)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"OCR failed: {exc}") from exc
    filename = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.jpg"
    cv2.imwrite(str(CAPTURES / filename), image)
    with closing(connect()) as db:
        cur = db.execute("INSERT INTO readings (meter_id,reading,confidence,image_name,roi_source,created_at) VALUES (?,?,?,?,?,?)",
                         (meter_id, result["reading"], result["confidence"], filename, result["roi"]["source"], int(time.time())))
        db.commit(); row_id = cur.lastrowid
    return {"ok": True, "id": row_id, "meter_id": meter_id, "image_url": f"/captures/{filename}", **result}


@app.post("/api/read")
async def receive_esp32(request: Request, meter_id: str = Query("meter-01"), x_meter_id: str | None = Header(None)):
    """ESP32 endpoint: POST raw JPEG bytes with Content-Type: image/jpeg."""
    data = await request.body()
    return JSONResponse(process_image(data, x_meter_id or meter_id))


@app.post("/api/upload")
async def upload_test_image(image: UploadFile = File(...), meter_id: str = Query("manual-upload")):
    return JSONResponse(process_image(await image.read(), meter_id))


def with_image_url(row: dict) -> dict:
    row["image_url"] = f"/captures/{row['image_name']}" if row.get("image_name") else ""
    return row


@app.get("/api/readings")
def readings(limit: int = Query(100, ge=1, le=1000)):
    with closing(connect()) as db:
        rows = [dict(row) for row in db.execute("SELECT * FROM readings WHERE batch_id IS NULL ORDER BY id DESC LIMIT ?", (limit,))]
    return [with_image_url(row) for row in rows]


@app.patch("/api/readings/{reading_id}")
def correct_reading(reading_id: int, reading: str = Body(embed=True)):
    cleaned = MeterReader.normalize(reading, int(reader.config().get("decimal_places", 0)))
    if not cleaned: raise HTTPException(400, "Reading must contain digits")
    with closing(connect()) as db:
        cur = db.execute("UPDATE readings SET reading=?, corrected=1 WHERE id=?", (cleaned, reading_id)); db.commit()
    if not cur.rowcount: raise HTTPException(404, "Reading not found")
    return {"ok": True, "reading": cleaned}


@app.get("/api/config")
def get_config(): return reader.config()


@app.put("/api/config")
def update_config(payload: dict = Body(...)):
    allowed = {k: payload[k] for k in ("roi", "decimal_places", "min_digits", "max_digits") if k in payload}
    return reader.save_config(allowed)


@app.get("/captures/{filename}")
def capture(filename: str):
    safe = Path(filename).name
    path = CAPTURES / safe
    if not path.exists(): raise HTTPException(404, "Image not found")
    return FileResponse(path)


# ---------------------------------------------------------------- dataset batches

def batch_row(db, batch_id: int) -> dict:
    row = db.execute("SELECT * FROM batches WHERE id=?", (batch_id,)).fetchone()
    if not row: raise HTTPException(404, "Batch not found")
    return dict(row)


def export_rows(batch_id: int | None) -> list[dict]:
    query = ("SELECT id,batch_id,meter_id,source_name,reading,confidence,expected,corrected,roi_source,note,"
             "datetime(created_at,'unixepoch','localtime') AS captured_at,image_name,created_at FROM readings ")
    with closing(connect()) as db:
        if batch_id is None:
            rows = db.execute(query + "WHERE batch_id IS NULL ORDER BY id").fetchall()
        else:
            batch_row(db, batch_id)
            rows = db.execute(query + "WHERE batch_id=? ORDER BY id", (batch_id,)).fetchall()
    split = ds.load_split(MODELS)
    return [ds.enrich(dict(row), split) for row in rows]


def save_preview(image, batch_id: int) -> str:
    """Batch photos are kept as small previews so a 1,000-image dataset does not fill the disk."""
    scale = 900 / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    filename = f"batch{batch_id}_{uuid.uuid4().hex[:10]}.jpg"
    cv2.imwrite(str(CAPTURES / filename), image, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return filename


def run_batch(batch_id: int, folder: Path, meter_id: str):
    cancel = cancel_flags[batch_id]
    processed = failed = 0
    try:
        with ds.Dataset(folder) as dataset, closing(connect()) as db:
            db.execute("UPDATE batches SET status='running' WHERE id=?", (batch_id,)); db.commit()
            for name, load in dataset:
                if cancel.is_set():
                    break
                expected = ds.expected_for(name, dataset.labels)
                reading, confidence, roi_source, image_name, note = "", 0.0, "", "", None
                try:
                    image = reader.decode(load())
                    result = reader.read(image)
                    reading, confidence, roi_source = result["reading"], result["confidence"], result["roi"]["source"]
                    image_name = save_preview(image, batch_id)
                except Exception as exc:  # a broken file is recorded, not fatal
                    note, failed = str(exc) or exc.__class__.__name__, failed + 1
                processed += 1
                db.execute("INSERT INTO readings (meter_id,reading,confidence,image_name,roi_source,created_at,batch_id,source_name,expected,note) "
                           "VALUES (?,?,?,?,?,?,?,?,?,?)",
                           (meter_id, reading, confidence, image_name, roi_source, int(time.time()), batch_id, name, expected, note))
                db.execute("UPDATE batches SET processed=?, failed=? WHERE id=?", (processed, failed, batch_id))
                db.commit()
            status = "cancelled" if cancel.is_set() else "done"
            db.execute("UPDATE batches SET status=?, finished_at=? WHERE id=?", (status, int(time.time()), batch_id)); db.commit()
    except Exception:
        with closing(connect()) as db:
            db.execute("UPDATE batches SET status='failed', finished_at=? WHERE id=?", (int(time.time()), batch_id)); db.commit()
        raise
    finally:
        cancel_flags.pop(batch_id, None)
        shutil.rmtree(folder, ignore_errors=True)


def safe_relative(name: str) -> Path:
    parts = [re.sub(r"[^\w.\- ]", "_", part)[:120] for part in name.replace("\\", "/").split("/") if part not in {"", ".", ".."}]
    return Path(*parts) if parts else Path(f"upload_{uuid.uuid4().hex[:8]}")


@app.post("/api/batches")
async def create_batch(request: Request, meter_id: str = Query("dataset"), name: str = Query("")):
    """Multipart field `files`: images, a ZIP, and/or a labels CSV. Folder uploads keep their relative paths."""
    form = await request.form(max_files=100_000, max_fields=100)
    uploads = [f for f in form.getlist("files") if isinstance(f, StarletteUploadFile)]
    if not uploads:
        raise HTTPException(400, "Choose a ZIP, a folder or some images")
    folder = BATCH_UPLOADS / uuid.uuid4().hex
    try:
        for upload in uploads:
            target = folder / safe_relative(upload.filename or "")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as out:
                shutil.copyfileobj(upload.file, out, 1024 * 1024)
            await upload.close()
        with ds.Dataset(folder) as dataset:
            total = len(dataset)
            labelled = sum(1 for n in dataset.names if ds.expected_for(n, dataset.labels))
    except Exception as exc:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(400, f"Could not open the upload: {exc}") from exc
    if not total:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(400, "No meter photos (.jpg, .png, .bmp, .webp) were found in the upload")
    first = safe_relative(uploads[0].filename or "")
    title = name.strip() or (first.parts[0] if len(first.parts) > 1 else first.stem if len(uploads) == 1 else f"{total} images")
    with closing(connect()) as db:
        cur = db.execute("INSERT INTO batches (name,meter_id,status,total,labelled,created_at) VALUES (?,?,?,?,?,?)",
                         (title[:120], meter_id[:60] or "dataset", "queued", total, labelled, int(time.time())))
        db.commit(); batch_id = cur.lastrowid
    cancel_flags[batch_id] = Event()
    batch_worker.submit(run_batch, batch_id, folder, meter_id[:60] or "dataset")
    return {"id": batch_id, "name": title, "total": total, "labelled": labelled, "status": "queued"}


@app.get("/api/batches")
def list_batches(limit: int = Query(20, ge=1, le=200)):
    with closing(connect()) as db:
        rows = [dict(r) for r in db.execute("SELECT * FROM batches ORDER BY id DESC LIMIT ?", (limit,))]
        for row in rows:
            stats = db.execute("SELECT COUNT(*) AS n, SUM(expected IS NOT NULL AND expected!='') AS labelled, "
                               "SUM(expected IS NOT NULL AND expected!='' AND reading!='' AND ABS(CAST(reading AS REAL)-CAST(expected AS REAL))<0.000001) AS matched "
                               "FROM readings WHERE batch_id=?", (row["id"],)).fetchone()
            row["accuracy"] = round(100 * (stats["matched"] or 0) / stats["labelled"], 1) if stats["labelled"] else None
    return rows


@app.get("/api/batches/{batch_id}")
def get_batch(batch_id: int):
    with closing(connect()) as db:
        batch = batch_row(db, batch_id)
    rows = export_rows(batch_id)
    batch["summary"] = [{"label": label, "value": value} for label, value in ds.summarize(rows)]
    return batch


@app.get("/api/batches/{batch_id}/readings")
def batch_readings(batch_id: int, show: str = Query("all", pattern="^(all|mismatch|empty|low)$"), limit: int = Query(200, ge=1, le=5000)):
    rows = export_rows(batch_id)
    if show == "mismatch":
        rows = [r for r in rows if r["match"] == "No"]
    elif show == "empty":
        rows = [r for r in rows if not r["reading"]]
    elif show == "low":
        rows = [r for r in rows if float(r["confidence"] or 0) < ds.LOW_CONFIDENCE]
    return [with_image_url(r) for r in reversed(rows)][:limit]


@app.post("/api/batches/{batch_id}/cancel")
def cancel_batch(batch_id: int):
    with closing(connect()) as db:
        batch_row(db, batch_id)
    flag = cancel_flags.get(batch_id)
    if flag: flag.set()
    return {"ok": True, "cancelling": bool(flag)}


@app.delete("/api/batches/{batch_id}")
def delete_batch(batch_id: int):
    with closing(connect()) as db:
        if batch_row(db, batch_id)["status"] in ("queued", "running"):
            raise HTTPException(409, "Cancel the batch before deleting it")
        for (image_name,) in db.execute("SELECT image_name FROM readings WHERE batch_id=? AND image_name!=''", (batch_id,)).fetchall():
            (CAPTURES / Path(image_name).name).unlink(missing_ok=True)
        db.execute("DELETE FROM readings WHERE batch_id=?", (batch_id,))
        db.execute("DELETE FROM batches WHERE id=?", (batch_id,)); db.commit()
    return {"ok": True}


def export_name(batch_id: int | None) -> str:
    if batch_id is None:
        return "water-meter-readings"
    with closing(connect()) as db:
        name = batch_row(db, batch_id)["name"]
    return f"batch-{batch_id}-" + (re.sub(r"[^\w\-]+", "-", name).strip("-")[:60] or "readings")


@app.get("/api/export.csv")
def export_csv(batch_id: int | None = Query(None)):
    rows = export_rows(batch_id)
    return StreamingResponse(iter([ds.csv_text(rows)]), media_type="text/csv",
                             headers={"Content-Disposition": f"attachment; filename={export_name(batch_id)}.csv"})


@app.get("/api/export.xlsx")
def export_xlsx(batch_id: int | None = Query(None)):
    rows = export_rows(batch_id)
    name = export_name(batch_id)
    content = ds.xlsx_bytes(rows, title=f"Water meter readings — {name}")
    return Response(content, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f"attachment; filename={name}.xlsx"})


@app.get("/health")
def health(): return {"ok": True, "service": "water-meter-reader"}
