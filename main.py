from fastapi import FastAPI
from pydantic import BaseModel
from rag import ask_rag
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_credentials=True,
    allow_methods=["*"],   
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Question):
    try:
        answer = ask_rag(q.question)
        return {"answer": answer}
    except Exception as e:
        import traceback
        error_msg = f"🐧 Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)  # log ใน terminal
        return {"answer": error_msg}  # ส่ง error กลับไปให้เห็นใน browser