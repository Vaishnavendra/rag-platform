from typing import Dict, List, Any,Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import uuid

from app.services.vector_store import vector_store_service
from app.services.llm import llm_service
from app.services.memory import memory_service

router=APIRouter(prefix="/genrate_answer",tags=["RAG Generation"])

class GenerateRequest(BaseModel):
    query:str
    session_id: Optional[str]= None
    top_k: int = 3

class GenerateResponse(BaseModel):
    session_id: str
    query:str
    answer:str
    sources: List[Dict[str,Any]]

@router.post("/answer",response_model=GenerateResponse)
async def generate_rag_answers(request: GenerateRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400,detail="Query string cant be empty")


    session_id=request.session_id or str(uuid.uuid4())

    history= memory_service.get_history(session_id)

    standalone_query= llm_service.contextualize_query(request.query,history)

#for searching relavent content
    retrived_chunks=vector_store_service.hybrid_search(
        query=standalone_query,
        limit=request.top_k
    )

#for answering 
    answer=llm_service.generate_answer(
        query=standalone_query,
        context_chunks=retrived_chunks
    )

    memory_service.add_message(session_id,"user",request.query)
    memory_service.add_message(session_id, "assistant",answer)

    return GenerateResponse(
        query=request.query,
        session_id=session_id,
        standalone_query=standalone_query,
        answer=answer,
        sources=retrived_chunks
    )