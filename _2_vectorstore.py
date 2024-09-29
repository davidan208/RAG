from typing import Union
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class VectorDB:
    def __init__(self, chunks = None, model_link: str = "dangvantuan/vietnamese-embedding-LongContext", vector_db: Union[Chroma, FAISS] = Chroma) -> None:
        self.model = HuggingFaceEmbeddings(model_name = model_link)
        self.vector_db  = vector_db
        self.db = self._build_db()

    def _build_db(self, documents):
        db = self.vector_db.from_documents(documents= documents, embedding = self.model)
        return db
    
    def get_retriever(self, search_type: str = "similarity", search_kwargs: dict = {"k" : 10}):
        retriever = self.db.as_retriever(search_type = search_type, search_kwargs = search_kwargs)
        return retriever
    
