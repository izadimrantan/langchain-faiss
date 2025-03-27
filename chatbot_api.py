from fastapi import FastAPI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
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
vector_db = FAISS.load_local("faiss_index", embeddings)

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
