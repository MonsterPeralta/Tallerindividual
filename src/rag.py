from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import os

class RAGSystem:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.vector_db = None
    
    def process_pdf(self, file_path):
        """Carga y procesa el PDF"""
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        
        # Dividir en chunks
        chunks = self.text_splitter.split_documents(pages)
        
        # Crear base de datos vectorial
        self.vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        return len(chunks)
    
    def query(self, question, k=3):
        """Consulta al sistema RAG"""
        if not self.vector_db:
            raise ValueError("Primero debe cargar un documento PDF")
            
        # Recuperar chunks relevantes
        docs = self.vector_db.similarity_search(question, k=k)
        return docs