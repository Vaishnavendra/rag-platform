import time
from typing import Tuple
from fastapi import HTTPException

def validate_query_gaurdrails(query:str)->str:
    cleaned=query.strip()
    if not cleaned:
        raise HTTPException(status_code=400,detail="Query can't be empty or left blank")
    if len(cleaned) < 3:
        raise HTTPException(status_code=400, detail="Query too short ")    
    if len(cleaned) > 1000:
        raise HTTPException(status_code=400, detail="Query too long ")

    return cleaned

class Timer:
    def __enter__(self):
        self.start=time.perf_counter()
        return self
    def __exit__(self, exc_type, exc, tb):
        self.end = time.perf_counter()
        self.interval_ms = round((self.end-self.start)*1000,2)