from typing import List,Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextChunkingService:
    def __init__(self,chunk_size: int = 500,chunk_overlap: int = 50):
        self.splitter= RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n","\n"," "]
        )

    def split_text(self,text : str) -> List[Dict[str,str]]:
        if not text.split():
            return []

        clean_text=text.replace("\r\n","\n").replace("\r","\n").replace("\t"," ")
        raw_chuncks= self.splitter.split_text(clean_text)
        structured_chunks=[]
        for index,chunk in enumerate(raw_chuncks):
            structured_chunks.append({
                "chunk_id": index,
                "content":chunk.strip(),
                "character_count":len(chunk.strip())
            })

        return structured_chunks

chunking_service=TextChunkingService()


        