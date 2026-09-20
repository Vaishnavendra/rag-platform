from typing import List , Dict , Any
from fastapi import APIRouter, Query, HTTPException
from app.services.vector_store import vector_store_service

router=APIRouter(prefix="/search",tags=["vector search"])

@router.get("/hybrid")
async def search_hybrid(
     query: str = Query(...,min_length=1,description="search query string"),
     limit: int = Query(default=3,ge=1,le=10,description="Max results to return"))-> Dict[str,Any]:
    if not query.strip():
        raise HTTPException(status_code= 400,detail="query cant be empty")

    results=vector_store_service.hybrid_search(query=query,limit=limit)
    return{
        "query":query,
        "total_results":len(results),
        "results":results
    }