from typing import List, Dict, Any
from groq import Groq
from app.core.config import settings

class llmservice():
    def __init__(self):
        self.client=Groq(api_key=settings.GROQ_API_KEY)
        self.model_name="openai/gpt-oss-20b" 

    def contextualize_query(self,query:str,history: List[Dict[str,str]]) -> str:
        if not history:
            return query
        formatted_history="\n".join([f"{msg['role'].upper()}:{msg['content']}"for msg in history])
        prompt=f"""Given the following conversation history and a follow-up question, rewrite the follow-up question to be a standalone question that can be understood WITHOUT the conversation history. Do NOT answer the question, just rewrite it if necessary.

CONVERSATION HISTORY:
{formatted_history}

FOLLOW-UP QUESTION:
{query}

STANDALONE QUESTION:"""
        response=self.client.chat.completions.create(
            messages=[{"role":"user","content":prompt}],
            model=self.model_name,
            temperature=0.2,
            max_tokens=100 
        )
        raw_output = response.choices[0].message.content.strip()
        cleaned_query=raw_output.replace("STANDALONE QUESTION:","").strip()
        return cleaned_query
    
        
    def build_prompt(self, query: str,context_chunks: List[Dict[str,Any]])->str:
        formatted_context=""
        for i, chunks in enumerate(context_chunks,1):
            formatted_context += f"\n--- Source Document[{i}]: chunk{chunks['document_title']} ---\n"
            formatted_context += f"{chunks['content']}\n"
        prompt=f"""You are a precise technical AI assistant. Answer the user's question strictly using only the provided context below.
                CRITICAL RULES:
                1. Do NOT use outside knowledge or speculate beyond the provided context.
                2. If the context does not contain enough information to answer the question, clearly state: "I cannot answer this question based on the provided context."
                3. Keep your answer concise, direct, and factual.

                CONTEXT:
                {formatted_context}

                USER QUESTION:
                {query}

                 ANSWER:"""

        return prompt

    def generate_answer(self, query: str, context_chunks: List[Dict[str,Any]]) -> str:
        if not context_chunks:
            return "No relevant context was available in the vector database for your question "

        formatted_prompt=self.build_prompt(query,context_chunks)

        response=self.client.chat.completions.create(
            messages=[
                {
                    "role":"user",
                    "content":formatted_prompt
                }
            ],
            model=self.model_name,
            temperature=0.1, #low means less hallucinations
            max_tokens=500
        )

        return response.choices[0].message.content.strip()

llm_service=llmservice()
