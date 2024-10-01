from fastapi import FastAPI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = ""  # YOUR OPENAI API KEY


app = FastAPI()

from _2_vectorstore import VectorDB
from fastapi import Request

vector_db = VectorDB()
retriever = vector_db.get_retriever()

llm = ChatOpenAI(model="gpt-4o-mini")


@app.get("/")
def root():
    return {"message": "Hello, World!"}


@app.post("/question")
async def rag(request: Request):
    query = await request.json()
    question = query.get("question")

    # Use RAG from retriever
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(question)
