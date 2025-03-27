import os
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Check if FAISS index exists
if not os.path.exists("faiss_index/index.faiss"):
    print("❌ FAISS index not found! Run Step 3 again.")
else:
    # Load FAISS vector database
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_db = FAISS.load_local("faiss_index", embeddings)

    # Get stored documents
    stored_docs = vector_db.docstore._dict

    print(f"✅ FAISS index loaded successfully!")
    print(f"🔢 Number of stored document chunks: {len(stored_docs)}")

    # Print first 100 stored text chunks
    for i, (key, doc) in enumerate(stored_docs.items()):
        print(f"\n🔹 Document {i+1}: {doc.page_content}")
        if i >= 100:  # Show only first 100
            break
