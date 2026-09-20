import uuid 
from typing import List,Dict,Any
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from fastembed import TextEmbedding

class VectorStoreService:
    def __init__(self,collection_name:str="rag_documents"):
        self.collection_name= collection_name
        self.client=QdrantClient(path="./qdrant-data")
        self.embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self._ensure_collection_exists()

    def _ensure_collection_exists(self) -> None:
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections) 

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE
                )
            )

    def generate_embeddigns(self,texts: List[str]) -> List[List[float]]:
        embeddings_generator = self.embedding_model.embed(texts)
        return [embedding.tolist() for embedding in embeddings_generator]

    def upsert_chunks(self,document_title: str, chunks: List[Dict[str, Any ]]) -> int:
        if not chunks:
            return 0

        texts= [chunk["content"] for chunk in chunks]
        vectors=self.generate_embeddigns(texts)

        points=[]
        for chunk,vector in zip(chunks,vectors):
            point_id = str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "document_title":document_title,"chunk_id":chunk["chunk_id"],
                        "content": chunk["content"],
                        "character_count": chunk["character_count"]
                }
            )
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )   
        return len(points)
    
vector_store_service = VectorStoreService()