import os
import faiss
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PDF_FILENAME = ""
# Step 1: Load Documentation (PDF)
loader = PyPDFLoader(PDF_FILENAME)  # Replace with your file
documents = loader.load()

# Step 2: Split Text into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Step 3: Convert Text to Vector Embeddings
embeddings = OpenAIEmbeddings()

# Step 4: Store Embeddings in FAISS
vector_db = FAISS.from_documents(docs, embeddings)

# Step 5: Save FAISS index
vector_db.save_local("faiss_index")

print("✅ Documentation successfully processed and stored in FAISS!")
