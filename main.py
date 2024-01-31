from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pathlib import Path
from starlette.requests import Request
from starlette.templating import Jinja2Templates
import shutil
import os
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, Form, File
from typing import List
from starlette.requests import Request
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter


app = FastAPI()
templates = Jinja2Templates(directory="templates")  # Directory containing your HTML templates


class Question(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str

db = None

# ---------------------------------------->
##########################################
openai_api_key = "YOUR OPENAI API KEY"
##########################################
#---------------------------------------->

# Directory to store uploaded files
UPLOAD_DIR = "uploads"
# Create the upload directory if it doesn't exist
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

# Initializing OpenAI Embeddings 
embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)



@app.get("/", response_class=HTMLResponse)
async def get_upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/upload/")
async def upload_files(request: Request, files: List[UploadFile] = File(...)):
    # Remove all previous files in the UPLOAD_DIR
    shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

    for file in files:
        with open(os.path.join(UPLOAD_DIR, file.filename), "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    global db
    db = generate_embedding(UPLOAD_DIR)
    
    return templates.TemplateResponse("chat.html", {"request": request})


def generate_embedding(upload_dir):
    # Your logic to generate embeddings
    from langchain.document_loaders import PyPDFLoader
    from langchain.document_loaders import DirectoryLoader

    loader = DirectoryLoader(UPLOAD_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    try:
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
        )
    except:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=200
        )
    docs = text_splitter.split_documents(docs)
    
    #Creating Embeddings
    db = FAISS.from_documents(docs, embeddings_model)
    return db

def conversation_chain(memory, retriever, openai_api_key):
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=OpenAI(openai_api_key=openai_api_key),
            retriever=retriever,
            memory=memory,
        )
        
        return conversation_chain

@app.post("/ask", response_model=Answer)
async def ask_question(item: Question):
    try:

        # Adding Memory To the Chatbot
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            )
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=OpenAI(openai_api_key=openai_api_key),
            retriever=db.as_retriever(),
            memory=memory,
        )
        
        # Initializing the converstational chain
        chat_history = []

        result = conversation_chain(
            {
                "question": item.question,
                "chat_history": chat_history
            }
        )

        chat_history = [
            (
                item.question,
                result["answer"]
            )
        ]
        answer = result["answer"]
        return {"answer": answer}
    except Exception as e:
        print(f"Error in ask_question: {str(e)}")  # For debugging
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing the question."
            )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

