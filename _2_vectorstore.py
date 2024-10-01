from typing import Union
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings


class VectorDB:
    def __init__(
        self,
        file_name=None,
        store_name=None,
        model_link: str = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
        vector_db: Union[Chroma, FAISS] = Chroma,
    ) -> None:
        # self.model = HuggingFaceEmbeddings(model_name=model_link)       # to large
        self.model = OpenAIEmbeddings()
        self.vector_db = vector_db
        self.file_name = file_name
        if store_name is None:
            loader = JSONLoader(
                file_path="/Users/duc.bui/Documents/BHD/test/RAG/json_src/test.json",
                jq_schema=".data[].content",
                text_content=False,
            )

            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )

            splits = text_splitter.split_documents(docs)
            self.db = self._build_db(splits)
        else:
            self.db = self.vector_db(
                persist_directory=store_name, embedding_function=self.model
            )

    def _build_db(self, documents):
        print("Building db")
        db = self.vector_db.from_documents(
            documents=documents,
            embedding=self.model,
            persist_directory="./chroma_langchain_db",
        )
        return db

    def get_retriever(
        self,
        search_type: str = "similarity_score_threshold",
        search_kwargs: dict = {"k": 4, "score_threshold": 0.7},
    ):
        retriever = self.db.as_retriever()
        return retriever
