from typing import List,Dict,Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.chunking import chunking_service
from app.services.vector_store import vector_store_service

router = APIRouter(prefix="/ingest",tags=["Document_ingestion"])

class IngestRequest(BaseModel):
    document_title : str
    text: str

@router.post("/process-text")
async def process_text(payload: IngestRequest) -> Dict[str,Any]:
    if not payload.text.strip():
        raise HTTPException(status_code=400,detail="Document Text cant be empty.")
    chunks=chunking_service.split_text(payload.text)

    chunks=chunking_service.split_text(payload.text)
    indexed_count=vector_store_service.upsert_chunks(
        document_title=payload.document_title,
        chunks=chunks
    )

    return { 
            "document_title":payload.document_title,
            "total_chunks": len(chunks),
            "indexed_vectors":indexed_count,
            "chunks":chunks
        }

