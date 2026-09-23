from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.services.chunking import chunking_service
from app.services.vector_store import vector_store_service

router = APIRouter(prefix="/ingest", tags=["Document_ingestion"])

class IngestRequest(BaseModel):
    document_title: Optional[str] = None
    title: Optional[str] = None
    text: Optional[str] = None
    content: Optional[str] = None

@router.post("/text")
@router.post("/process-text")
async def process_text(payload: IngestRequest) -> Dict[str, Any]:
    title = payload.title or payload.document_title or "Untitled Document"
    raw_text = payload.content or payload.text or ""

    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="Document Text cant be empty.")

    chunks = chunking_service.split_text(raw_text)
    indexed_count = vector_store_service.upsert_chunks(
        document_title=title,
        chunks=chunks
    )

    return {
        "document_title": title,
        "title": title,
        "total_chunks": len(chunks),
        "indexed_vectors": indexed_count,
        "chunks": chunks
    }
