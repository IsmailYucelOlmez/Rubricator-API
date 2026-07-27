import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers.semantic import router as semantic_router
from app.api.routers.sessions import router as sessions_router
from app.core.config import settings
from app.data.session_store import get_session_store

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_session_store().start_cleanup_task()
    yield
    get_session_store().stop_cleanup_task()


app = FastAPI(title="BookApp Semantic API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(semantic_router)
app.include_router(sessions_router)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "bookapp-api", "status": "ok"}
