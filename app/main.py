"""Entry point FastAPI.

Jalankan (dari root project):
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --ssl-keyfile key.pem --ssl-certfile cert.pem
PENTING: gunakan 1 worker (default). Multi-worker = beberapa proses = beberapa model di VRAM -> OOM.
"""
from __future__ import annotations

import hmac
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import routes_doctors, routes_jobs, routes_misc
from app.config import get_settings
from app.core.services import build_services

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")
STATIC_DIR = Path(__file__).parent / "static"
PAGES = {"/": "index.html", "/enroll": "enroll.html", "/review": "review.html",
         "/metrics": "metrics.html", "/cases": "cases.html"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    services = build_services(settings)
    app.state.services = services
    await services.queue.start()
    # Recovery: job 'queued' dimasukkan ulang; job 'processing' saat server mati -> failed.
    for job in services.repo.jobs_with_status(("queued", "processing")):
        if job["status"] == "queued":
            await services.queue.submit(job["id"])
        else:
            services.repo.update_job(job["id"], status="failed", stage="failed",
                                     error="Server berhenti saat job diproses. Unggah ulang rekaman.")
    log.info("Server siap. LLM=%s Whisper=%s swap=%s", settings.llm_backend, settings.whisper_model,
             settings.gpu_swap_policy)
    yield
    await services.queue.stop()
    services.ctx.slot.unload_all()


app = FastAPI(title="Offline Medical STT & OSCE Case Bank", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def token_auth(request: Request, call_next):
    token = request.app.state.services.settings.access_token if hasattr(request.app.state, "services") else ""
    path = request.url.path
    if token and path.startswith("/api/") and path != "/api/health":
        supplied = (request.headers.get("x-access-token") or request.cookies.get("access_token")
                    or request.query_params.get("token") or "")
        if not hmac.compare_digest(supplied.encode(), token.encode()):
            return JSONResponse({"detail": "Token akses tidak valid."}, status_code=401)
    return await call_next(request)


app.include_router(routes_misc.router)
app.include_router(routes_doctors.router)
app.include_router(routes_jobs.router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _page_handler(filename: str):
    async def handler() -> FileResponse:
        return FileResponse(STATIC_DIR / filename, headers={"Cache-Control": "no-cache"})
    return handler


for route, filename in PAGES.items():
    app.add_api_route(route, _page_handler(filename), include_in_schema=False, methods=["GET"])
