"""Document loading utilities for various file formats."""

import os
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader
)
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .config import Config


class DocumentLoader:
    """Handles loading and processing of various document types."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load a document based on its file extension."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = file_path.suffix.lower()
        
        if extension == '.txt' or extension == '.md':
            loader = TextLoader(str(file_path), encoding='utf-8')
        elif extension == '.pdf':
            loader = PyPDFLoader(str(file_path))
        elif extension == '.docx':
            loader = UnstructuredWordDocumentLoader(str(file_path))
        else:
            raise ValueError(f"Unsupported file type: {extension}")
        
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
    
    def load_text(self, text: str) -> List[Document]:
        """Load text directly as documents."""
        documents = [Document(page_content=text, metadata={"source": "direct_input"})]
        return self.text_splitter.split_documents(documents)
    
    def load_multiple_files(self, file_paths: List[str]) -> List[Document]:
        """Load multiple files and combine them."""
        all_documents = []
        for file_path in file_paths:
            try:
                documents = self.load_document(file_path)
                all_documents.extend(documents)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        return all_documents
