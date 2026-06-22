import streamlit as st
import requests
import os

# 1. IMMEDIATE PAGE SETUP (Makes the UI render instantly)
st.set_page_config(page_title="Enterprise RAG Dashboard", page_icon="🤖", layout="wide")

st.title("🤖 Custom AI Knowledge Base Chatbot")
st.caption("Upload documents in real-time and query them using hybrid vector-retrained intelligence.")
st.write("---")

API_URL = "http://127.0.0.1:8000/chat"

# 2. LAZY-LOAD HEAVY ML LIBRARIES ONLY WHEN NEEDED
def process_uploaded_file(uploaded_file, temp_path):
    # These massive imports happen inside the function, hiding them from Streamlit's startup loop
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    import chromadb
    from chromadb.utils import embedding_functions
    import time

    # Initialize Chroma Client
    DB_PATH = "./chroma_db"
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    # Configure Docling Ingestion Options
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_layout = True
    converter = DocumentConverter(format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_options)})
    
    # Convert document layers
    result = converter.convert(temp_path)
    
    chunks = []
    metas = []
    ids = []
    
    # Extract structural components
    for i, element in enumerate(result.document.export_to_markdown().split("\n\n")):
        if element.strip():
            chunks.append(element)
            metas.append({"source": uploaded_file.name, "page_number": 1})
            ids.append(f"doc_{int(time.time())}_{i}")
    
    # Refresh Active Vector Space
    try:
        chroma_client.delete_collection(name="company_knowledge_base")
    except Exception:
        pass
    
    collection = chroma_client.create_collection(
        name="company_knowledge_base", 
        embedding_function=embedding_func
    )
    
    collection.add(documents=chunks, metadatas=metas, ids=ids)

# 3. SIDEBAR FOR DOCUMENT MANAGEMENT
with st.sidebar:
    st.header("📄 Knowledge Ingestion")
    st.write("Upload a new PDF to clear the active database and train the assistant on new data.")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        if "processed_file" not in st.session_state or st.session_state.processed_file != uploaded_file.name:
            with st.status(f"Processing '{uploaded_file.name}'...", expanded=True) as status:
                try:
                    os.makedirs("temp_data", exist_ok=True)
                    temp_path = os.path.join("temp_data", uploaded_file.name)
                    
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    status.write("🔍 Running Layout & Text Analytics...")
                    
                    # Execute lazy-loaded pipeline
                    process_uploaded_file(uploaded_file, temp_path)
                    
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                    st.session_state.processed_file = uploaded_file.name
                    status.update(label="✅ Ingestion Successful!", state="complete", expanded=False)
                    st.success(f"Successfully trained on {uploaded_file.name}!")
                    st.rerun()
                    
                except Exception as e:
                    status.update(label="❌ Ingestion Failed", state="error")
                    st.error(f"Error processing file: {str(e)}")

# 4. CONVERSATIONAL SYSTEM WINDOW
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("🔍 Citations Used"):
                for src in message["sources"]:
                    st.write(f"- **File:** {src['file']} | **Page:** {src['page']}")

if user_query := st.chat_input("Ask something about the uploaded document..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching document data layers..."):
            try:
                response = requests.post(API_URL, json={"question": user_query})
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data["sources"]
                    
                    st.markdown(answer)
                    if sources:
                        with st.expander("🔍 Citations Used"):
                            unique_sources = {f"{s['file']}_p{s['page']}": s for s in sources}.values()
                            for src in unique_sources:
                                f_name = src['file'] if src['file'] else st.session_state.get('processed_file', 'Uploaded File')
                                st.write(f"- **File:** {f_name} | **Page:** {src['page']}")
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
                else:
                    st.error(f"Backend returned error: {response.status_code}")
            except Exception as e:
                st.error(f"Could not connect to backend engine: {str(e)}")