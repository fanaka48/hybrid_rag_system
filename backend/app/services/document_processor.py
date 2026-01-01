import os
import pickle
import asyncio
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document as LCDocument
from app.core.config import settings
from loguru import logger

class DocumentProcessor:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        self.vector_store_path = settings.FAISS_INDEX_PATH
        self.bm25_path = os.path.join(settings.STORAGE_PATH, "bm25.pkl")

    async def process_file(self, file_path: str, metadata: dict) -> List[LCDocument]:
        return await asyncio.to_thread(self._process_file_sync, file_path, metadata)

    def _process_file_sync(self, file_path: str, metadata: dict) -> List[LCDocument]:
        logger.info(f"Processing file: {file_path}")
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)
        
        docs = loader.load()
        for doc in docs:
            doc.metadata.update(metadata)
        
        chunks = self.text_splitter.split_documents(docs)
        logger.info(f"Created {len(chunks)} chunks from {file_path}")
        return chunks

    async def add_to_index(self, chunks: List[LCDocument]):
        await asyncio.to_thread(self._add_to_index_sync, chunks)

    def _add_to_index_sync(self, chunks: List[LCDocument]):
        # Add to FAISS
        if os.path.exists(self.vector_store_path):
            vector_db = FAISS.load_local(self.vector_store_path, self.embeddings, allow_dangerous_deserialization=True)
            vector_db.add_documents(chunks)
        else:
            vector_db = FAISS.from_documents(chunks, self.embeddings)
        
        vector_db.save_local(self.vector_store_path)
        
        # Update BM25
        self._update_bm25_sync(chunks)

    def _update_bm25_sync(self, new_chunks: List[LCDocument]):
        all_chunks = []
        if os.path.exists(self.bm25_path):
            with open(self.bm25_path, "rb") as f:
                all_chunks = pickle.load(f)
        
        all_chunks.extend(new_chunks)
        
        with open(self.bm25_path, "wb") as f:
            pickle.dump(all_chunks, f)
        
        logger.info("BM25 index updated")

    async def delete_document(self, doc_id: int):
        await asyncio.to_thread(self._delete_document_sync, doc_id)

    def _delete_document_sync(self, doc_id: int):
        logger.info(f"Deleting document chunks for doc_id: {doc_id}")
        
        # 1. Filter BM25 pkl
        all_chunks = []
        if os.path.exists(self.bm25_path):
            with open(self.bm25_path, "rb") as f:
                all_chunks = pickle.load(f)
        
        filtered_chunks = [c for c in all_chunks if c.metadata.get("doc_id") != doc_id]
        
        with open(self.bm25_path, "wb") as f:
            pickle.dump(filtered_chunks, f)
        
        # 2. Rebuild FAISS index
        if os.path.exists(self.vector_store_path):
            if filtered_chunks:
                vector_db = FAISS.from_documents(filtered_chunks, self.embeddings)
                vector_db.save_local(self.vector_store_path)
            else:
                import shutil
                shutil.rmtree(self.vector_store_path, ignore_errors=True)
        
        logger.info(f"Document {doc_id} removed from indices")

    async def get_retriever_data(self):
        return await asyncio.to_thread(self._get_retriever_data_sync)

    def _get_retriever_data_sync(self):
        vector_db = None
        if os.path.exists(self.vector_store_path):
            vector_db = FAISS.load_local(self.vector_store_path, self.embeddings, allow_dangerous_deserialization=True)
        
        bm25_chunks = []
        if os.path.exists(self.bm25_path):
            with open(self.bm25_path, "rb") as f:
                bm25_chunks = pickle.load(f)
        
        return vector_db, bm25_chunks

doc_processor = DocumentProcessor()
