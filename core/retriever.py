import os
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def setup_tool_retriever(all_tools):
    """Embeds tool descriptions and sends them to the ChromaDB Docker container."""
    print("Initializing local embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Extract names and descriptions for the vector store
    texts = [f"{t.name}: {t.description}" for t in all_tools]
    metadatas = [{"name": t.name} for t in all_tools]
    
    print("Connecting to ChromaDB Container...")
    # Fetch the host and port from the docker-compose environment variables
    chroma_host = os.getenv("CHROMA_HOST", "localhost")
    chroma_port = os.getenv("CHROMA_PORT", "8000")
    
    # Create the HTTP client to talk to the database container
    chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    
    print("Populating database...")
    vectorstore = Chroma.from_texts(
        client=chroma_client,
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name="agent_tools"
    )
    
    # Set the retriever to only grab the top 3 most relevant tools
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) 
    return retriever