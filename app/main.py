from fastapi import FastAPI
from app.core.config import settings
from app.api.v1.ingest import router as ingest_router
from app.api.v1.search import router as search_router
from app.api.v1.generate import router as generate_answer_router
app = FastAPI(
    title= "Distributed Real-Time RAG Platform",
    description="Trying to create a RAG backend with vector caching and bg workers",
    version="0.1.0"
)

app.include_router(ingest_router,prefix="/api/v1")
app.include_router(search_router,prefix="/api/v1")
app.include_router(generate_answer_router,prefix="/api/v1")

@app.get("/")
async def root():
    return{
        "message":f"{settings.PROJECT_NAME} API IS LIVE",
        "enviromnet": settings.ENVIRONMENT,
        "docs_url":"/docs"
    }

@app.get("/health")
async def health_check():
    return{
        "status":"healthy",
        "service":settings.PROJECT_NAME,
        "debug_mode":settings.DEBUG
    }

