from typing import Literal
import os
import multiprocessing
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader, DirectoryLoader
from langchain_core.document_loaders import BaseLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

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
        embeddings = HuggingFaceEmbeddings(model_name="VoVanPhuc/sup-SimCSE-VietNamese-phobert-base", cache_folder = "./embed_model")
        self.text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type = "standard_deviation")
    
    def __call__(self, link: str, **kwargs):
        pass
    
    def split_documents(self, docs):
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
    
class Loader:
    def __init__(self):
        # self.pdf_loader = PDFLoader()
        self.txt_loader = TXTLoader()
        # self.md_loader = MDLoader()
    def __call__(self, link: str, file_type: Literal['pdf', 'txt', 'md'], **kwargs):
        match file_type:
            case "txt":
                return self.txt_loader(link, **kwargs)
            case _:
                raise ValueError("[PROBLEM PENDING] Chưa hỗ trợ định dạng này")
        # if file_type == 'pdf':
        #     return self.pdf_loader(link, **kwargs)
        # elif file_type == 'txt':
        #     return self.txt_loader(link, **kwargs)
        # elif file_type == 'md':
        #     return self.md_loader(link, **kwargs)
        # else:
        #     raise ValueError("Unsupported file type")
        
a = Loader()
chunks = a('./src/sub.txt', 'txt')
print(len(chunks))