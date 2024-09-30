from typing import Literal
import os
import multiprocessing
from langchain_community.document_loaders import PyPDFLoader, TextLoader, JSONLoader, UnstructuredMarkdownLoader, DirectoryLoader
from langchain_core.document_loaders import BaseLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import json

def remove_non_utf8_characters(text):
    Vietnamese_character_ranges = [(32, 126), (160, 255), (256, 383), (384, 591), (768, 879), (7680, 7935)]
    def is_vietnamese_char(char):
        return any(start <= ord(char) <= end for start, end in Vietnamese_character_ranges)
    return ''.join(char for char in text if is_vietnamese_char(char))

def get_cpu():
    return multiprocessing.cpu_count()

class BaseLoader:
    def __init__(self) -> None:
        self.num_process = get_cpu()
        embeddings = HuggingFaceEmbeddings(model_name="dangvantuan/vietnamese-embedding-LongContext", cache_folder = "./embed_model", model_kwargs = {'trust_remote_code': True})
        self.text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type = "standard_deviation", breakpoint_threshold_amount = 4)
    
    def __call__(self, link: str, **kwargs):
        pass
    
    def split_document(self, docs):
        # Combine all documents into a single large document
        combined_text = "\n\n".join([doc.page_content for doc in docs])
        combined_doc = Document(page_content=combined_text, metadata={"source": "combined"})
        
        # Split the combined document
        chunks = self.text_splitter.split_documents([combined_doc])
        
        # Assign appropriate metadata to each chunk
        for chunk in chunks:
            chunk.metadata["original_sources"] = [doc.metadata.get("source") for doc in docs]
        
        return chunks

class PDFLoader(BaseLoader):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, link: str, **kwargs):
        if os.path.isfile(link):
            pdf_loader = PyPDFLoader(link)
            docs = pdf_loader.load()
        elif os.path.isdir(link):
            pdf_loader = DirectoryLoader(   path = link, 
                                            glob = "**/*.pdf", 
                                            loader_cls = PyPDFLoader, 
                                            show_progress = True, 
                                            use_multithreading = True, 
                                            max_concurrency = min(self.num_process, kwargs['workers'])  )
            docs = pdf_loader.load()
        
        for doc in docs:
            doc.page_content = remove_non_utf8_characters(doc.page_content)
        return self.split_documents(docs)
    
class TXTLoader(BaseLoader):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, link: str, **kwargs):
        if os.path.isfile(link):
            txt_loader = TextLoader(link)
            docs = txt_loader.load()
        elif os.path.isdir(link):
            txt_loader = DirectoryLoader(   path = link,
                                            glob = "**/*.txt",
                                            loader_cls = TextLoader,
                                            show_progress = True,
                                            use_multithreading = True,
                                            max_concurrency = min(self.num_process, kwargs['workers'])   )
            docs = txt_loader.load()

        for doc in docs:
            doc.page_content = remove_non_utf8_characters(doc.page_content)
        return self.split_documents(docs)

class MDLoader(BaseLoader):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, link: str, **kwargs):
        if os.path.isfile(link):
            md_loader = UnstructuredMarkdownLoader(link)
            docs = md_loader.load()
        elif os.path.isdir(link):
            md_loader = DirectoryLoader(    path = link,
                                            glob = "**/*.md",
                                            loader_cls = UnstructuredMarkdownLoader,
                                            show_progress = True,
                                            use_multithreading = True,
                                            max_concurrency = min(self.num_process, kwargs['workers']))
            docs = md_loader.load()
        
        return self.split_documents(docs)

class _JsonLoader(BaseLoader):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, link: str, **kwargs):
        if os.path.isfile(link):
            print("FOUND FILE")
            js_loader = JSONLoader( file_path = link,
                                    jq_schema = '.content')
        elif os.path.isdir(link):
            print("FOUND DIR")
            def custom_json_loader(file_path: str):
                return JSONLoader(file_path=file_path, jq_schema='.content')

            js_loader = DirectoryLoader(    path = link,
                                            glob = "**/*.json",
                                            loader_cls = custom_json_loader,
                                            show_progress = True,
                                            use_multithreading = True,
                                            max_concurrency = min(self.num_process, kwargs['workers']) )

        docs = js_loader.load()
        for doc in docs:
            doc.page_content = remove_non_utf8_characters(doc.page_content)
        return self.split_documents(docs)
    
class JsonLoader(BaseLoader):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, link: str, **kwargs):
        # Load JSON content from the link (or file path)
        with open(link, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        
        # Convert JSON content into documents, skipping those with empty or blank 'content'
        docs = [
            Document(page_content=remove_non_utf8_characters(item["content"]), metadata={"source": item["#url"], "title": item["title"], "description": item["description"], "date": item["meta_data"]["datetime_crawled"], "keywords": item["meta_data"]["keywords"]})
            for item in json_data
            if item["content"].strip()  # Skip empty or blank content
        ]
        
        # Split the documents into chunks
        chunks = self.split_document(docs)
        return chunks
    
    def split_document(self, docs):
        chunks = self.text_splitter.split_documents(docs)
        return chunks
 
class Loader:
    def __init__(self):
        self.json_loader = JsonLoader()
    def __call__(self, link: str, file_type: Literal['pdf', 'txt', 'md', 'json']):
        match file_type:
            case "json":
                return self.json_loader(link)
            case _:
                raise ValueError("[PROBLEM PENDING] Chưa hỗ trợ định dạng này")

import time    

start = time.time()
a = Loader()
b = a('./json_src/crawl_data_3009.json', 'json')
print(f"{time.time() - start}s for {len(b)}")
print(b[-2])
