from typing import Dict, List, Any,Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import uuid

from app.services.vector_store import vector_store_service
from app.services.llm import llm_service
from app.services.memory import memory_service
from app.core.gaurdrails import validate_query_gaurdrails,Timer

router=APIRouter(prefix="/genrate_answer",tags=["RAG Generation"])

class GenerateRequest(BaseModel):
    query:str
    session_id: Optional[str]= None
    top_k: int = 3


class LatencyMetrics(BaseModel):
    retrieval_ms:float
    llm_generation_ms:float
    total_ms: float

class GenerateResponse(BaseModel):
    query:str
    session_id: str
    standalone_query:str
    answer:str
    metrices:LatencyMetrics
    sources: List[Dict[str,Any]]

@router.post("/answer",response_model=GenerateResponse)
async def generate_rag_answers(request: GenerateRequest):
    with Timer() as total_timer:
        cleaned_query= validate_query_gaurdrails(request.query)

        session_id=request.session_id or str(uuid.uuid4())

        history= memory_service.get_history(session_id)

        standalone_query= llm_service.contextualize_query(request.query,history)

#for searching relavent content(module 7) // with timer (module 8)  
        with Timer() as retrieval_timer:
            retrived_chunks=vector_store_service.hybrid_search(
             query=standalone_query,
                limit=request.top_k
        )

#for answering (module 7) // with timer (module 8)  
        with Timer() as llm_timer:
            answer=llm_service.generate_answer(
                query=standalone_query,
                context_chunks=retrived_chunks
        )

        memory_service.add_message(session_id,"user",request.query)
        memory_service.add_message(session_id, "assistant",answer)

    return GenerateResponse(
        query=cleaned_query,
        session_id=session_id,
        standalone_query=standalone_query,
        answer=answer,
        metrices=LatencyMetrics(
            retrieval_ms=retrieval_timer.interval_ms,
            llm_generation_ms=llm_timer.interval_ms,
            total_ms=total_timer.interval_ms
        ),
        sources=retrived_chunks
        )