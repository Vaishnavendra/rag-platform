import uuid 
from typing import List,Dict,Any
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, SparseVectorParams, Modifier, Prefetch, FusionQuery, Fusion)
from fastembed import TextEmbedding,SparseTextEmbedding

class VectorStoreService:
    def __init__(self,collection_name:str="rag_documents"):
        self.collection_name= collection_name
        self.client=QdrantClient(path="./qdrant_data")

        self.dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

        self.sparse_model=SparseTextEmbedding(model_name="Qdrant/bm25")

        self._ensure_collection_exists()

    def _ensure_collection_exists(self) -> None:
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections) 

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense":VectorParams(size=384,distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse":SparseVectorParams(modifier=Modifier.IDF)
                }
            )

    def generate_embeddigns(self,texts: List[str]) -> List[List[float]]:
        embeddings_generator = self.embedding_model.embed(texts)
        return [embedding.tolist() for embedding in embeddings_generator]

    def upsert_chunks(self,document_title: str, chunks: List[Dict[str, Any ]]) -> int:
        if not chunks:
            return 0

        texts= [chunk["content"] for chunk in chunks]

        dense_vector=[e.tolist() for e in self.dense_model.embed(texts)]
        sparse_vector=list(self.sparse_model.embed(texts))

        points=[]
        for chunk,dense_vec, sparse_vec in zip(chunks,dense_vector,sparse_vector):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense":dense_vec,
                        "sparse":sparse_vec.as_object()
                    },
                    payload={
                        "document_title":document_title,
                        "chunk_id":chunk["chunk_id"],
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


    def hybrid_search(self,query:str,limit: int =3 ) ->List[Dict[str, Any]]:
        dense_query=list(self.dense_model.embed([query]))[0].tolist()
        sparse_query=list(self.sparse_model.embed([query]))[0].as_object()

        results=self.client .query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(query=dense_query,using="dense",limit=limit * 2),
                Prefetch(query=sparse_query,using="sparse",limit=limit * 2)
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit
        )

        return [
            {
                "score":point.score,
                "document_title":point.payload["document_title"],
                "chunk_id": point.payload["chunk_id"],
                "content": point.payload["content"]
            }
            for point in results.points
        ]

    
vector_store_service = VectorStoreService()