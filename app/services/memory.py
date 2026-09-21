from typing import List, Dict
from collections import defaultdict

class ConversationMemoryService:
    def __init__(self):
        self._session: Dict[str,List[Dict[str,str]]]=defaultdict(list)

    def add_message(self,session_id:str,role:str , content: str):
        self._session[session_id].append({"role":role,"content":content})

    def get_history(self,session_id : str, limit: int = 6) -> List[Dict[str,str]]:
        return self._session[session_id[-limit:]]

    def clear_history(self,session_id: str):
        if session_id in self._session:
            del self._session[session_id]


memory_service=ConversationMemoryService()