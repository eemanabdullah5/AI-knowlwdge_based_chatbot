import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
from google import genai
from dotenv import load_dotenv

# Load environment variables (.env file)
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="AI Knowledge Base RAG Backend")

# Initialize Gemini Client
# The SDK automatically looks for the GEMINI_API_KEY environment variable
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("❌ GEMINI_API_KEY not found in environment variables!")
ai_client = genai.Client()

# Initialize ChromaDB (Pointing to the database you just created in Phase 1)
DB_PATH = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=DB_PATH)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = chroma_client.get_collection(
    name="company_knowledge_base", 
    embedding_function=embedding_func
)

# Define what a user's incoming request looks like
class QueryRequest(BaseModel):
    question: str

# ==========================================
# CORE RAG API ENDPOINT
# ==========================================
@app.post("/chat")
async def chat_with_docs(request: QueryRequest):
    try:
        # 1. RETRIEVAL: Search ChromaDB for the top 3 matching chunks
        results = collection.query(
            query_texts=[request.question],
            n_results=3
        )
        
        # Flatten retrieved text chunks and keep track of document sources
        retrieved_chunks = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        if not retrieved_chunks:
            return {"answer": "I couldn't find any information regarding that in your documents.", "sources": []}
            
        # Combine the chunks into a unified string of data context
        context_str = "\n---\n".join(retrieved_chunks)
        
        # Capture source file names and pages for client transparency
        sources = [
            {"file": meta.get("source"), "page": meta.get("page_number", 1)} 
            for meta in metadatas
        ]
        
        # 2. GENERATION: Build a robust system prompt preventing hallucinations
        system_instruction = (
            "You are an expert AI Assistant specialized in analyzing uploaded corporate documentation.\n"
            "Your core rule: Answer the user's question using ONLY the provided document context below.\n"
            "If the answer cannot be found or reasonably inferred from the context, explicitly say: "
            "'I cannot find the answer to that within the provided documents.' Do not make things up."
        )
        
        user_prompt = f"DOCUMENT CONTEXT:\n{context_str}\n\nUSER QUESTION: {request.question}"
        
        # Send everything to Gemini
        response = ai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_prompt,
            config={"system_instruction": system_instruction}
        )
        
        # Return structured JSON response back to our future frontend UI
        return {
            "answer": response.text,
            "sources": sources
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run app locally via: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)