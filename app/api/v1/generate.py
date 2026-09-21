from typing import Dict, List, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.vector_store import vector_store_service
from app.services.llm import llm_service


router=APIRouter(prefix="/genrate_answer",tags=["RAG Generation"])

class GenerateRequest(BaseModel):
    query:str
    top_k: int = 3

class GenerateResponse(BaseModel):
    query:str
    answer:str
    sources: List[Dict[str,Any]]

@router.post("/answer",response_model=GenerateResponse)
async def generate_rag_answers(request: GenerateRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400,detail="Query string cant be empty")

#for searching relavent content
    retrived_chunks=vector_store_service.hybrid_search(
        query=request.query,
        limit=request.top_k
    )

#for answering 
    answer=llm_service.generate_answer(
        query=request.query,
        context_chunks=retrived_chunks
    )

    return GenerateResponse(
        query=request.query,
        answer=answer,
        sources=retrived_chunks
    )