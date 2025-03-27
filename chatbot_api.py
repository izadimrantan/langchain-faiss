from fastapi import FastAPI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv
from urllib.parse import quote

# Load environment variables
load_dotenv()

app = FastAPI()

# Load OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load FAISS Index
embeddings = OpenAIEmbeddings()
try:
    print("Starting chatbot...")
    print("Loading FAISS index...")
    vector_db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("FAISS index loaded.")
except Exception as e:
    print(f"Failed to load FAISS: {e}")

# Load OpenAI GPT-4 Model
llm = ChatOpenAI(model_name="gpt-4")

# Prompt Template for LLM
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="Context: {context}\n\nQuestion: {question}\nAnswer:"
)

@app.get("/ask")
def ask(query: str):
    # URL-encode the query to handle spaces and special characters
    encoded_query = quote(query)
    
    # Search for similar documents
    similar_docs = vector_db.similarity_search(query, k=3)
    
    # Extract text from retrieved documents
    context = "\n\n".join([doc.page_content for doc in similar_docs])

    # Generate response using LLM
    chain = LLMChain(llm=llm, prompt=prompt)
    answer = chain.run(context=context, question=query)
    
    return {"answer": answer}

@app.get("/ping")
def ping():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    port = int(os.environ.get("PORT", 8000))  # Use Railway's PORT or fallback to 8000 locally
    uvicorn.run(app, host="0.0.0.0", port=port)
