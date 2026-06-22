import os
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
import chromadb
from chromadb.utils import embedding_functions

# ==========================================
# 1. INITIALIZATION & CONFIGURATION
# ==========================================
# Initialize the Docling Converter (Handles PDFs, DOCX, PPTX, HTML, etc.)
converter = DocumentConverter()

# Initialize Local Vector Database
DB_PATH = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=DB_PATH)

# Use a high-quality local embedding model (or swap with OpenAI/Gemini API embedding)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create or fetch our collection
collection = chroma_client.get_or_create_collection(
    name="company_knowledge_base", 
    embedding_function=embedding_func
)

# ==========================================
# 2. CORE PROCESSING FUNCTIONS
# ==========================================
def process_document(file_path: str):
    """
    Parses a document, chunks it intelligently while preserving tables/layout,
    and returns text chunks along with metadata.
    """
    print(f"📄 Parsing document: {file_path}...")
    
    # Docling automatically detects structure, formats tables into Markdown,
    # and keeps columns reading in the correct order.
    result = converter.convert(file_path)
    docling_doc = result.document
    
    # Define a chunker (HybridChunker respects document boundaries like headers and tables)
    chunker = HybridChunker(max_tokens=400, merge_peers=True)
    chunk_iterable = chunker.chunk(docling_doc)
    
    documents = []
    metadatas = []
    ids = []
    
    filename = Path(file_path).name

    for index, chunk in enumerate(chunk_iterable):
        # Extract the text content of the chunk (tables will be clean Markdown string layouts)
        text_content = chunker.serialize(chunk)
        
        documents.append(text_content)
        ids.append(f"{filename}_chunk_{index}")
        
        # Meta-data tracking for source citations later
        metadatas.append({
            "source": filename,
            "chunk_index": index,
            # If docling successfully extracts page numbers, attach it here
            "page_number": getattr(chunk.meta, "page_no", 1) 
        })
        
    return documents, metadatas, ids

def load_to_vector_store(documents, metadatas, ids):
    """
    Upserts the parsed text chunks and metadata into ChromaDB.
    """
    if not documents:
        print("⚠️ No text extracted to load.")
        return
        
    print(f"📦 Indexing {len(documents)} chunks into ChromaDB...")
    
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print("✅ Ingestion successfully completed!")

# ==========================================
# 3. EXECUTION ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    # Test path - Replace with a sample PDF containing columns or a table
    SAMPLE_FILE = "data/sample_policy_or_report.pdf" 
    
    if os.path.exists(SAMPLE_FILE):
        docs, metas, chunk_ids = process_document(SAMPLE_FILE)
        load_to_vector_store(docs, metas, chunk_ids)
    else:
        print(f"❌ Target file not found at '{SAMPLE_FILE}'. Please place a file there to test.")
        