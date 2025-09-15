"""Basic text summarization using LangChain."""

from typing import List, Optional, Dict, Any
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from openai import OpenAI as OpenAIClient
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from .config import Config
from .document_loader import DocumentLoader


class BasicSummarizer:
    """Basic text summarizer using LangChain chains."""
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize the summarizer with specified model."""
        Config.validate_config()
        
        self.model_name = model_name or Config.DEFAULT_MODEL
        
        # Use Perplexity API with OpenAI-compatible interface
        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS,
            openai_api_key=Config.PERPLEXITY_API_KEY,
            openai_api_base=Config.PERPLEXITY_BASE_URL
        )
        self.document_loader = DocumentLoader()
    
    def summarize_text(self, text: str, summary_type: str = "stuff") -> str:
        """Summarize a text string directly."""
        documents = self.document_loader.load_text(text)
        return self._summarize_documents(documents, summary_type)
    
    def summarize_file(self, file_path: str, summary_type: str = "map_reduce") -> str:
        """Summarize a file."""
        documents = self.document_loader.load_document(file_path)
        return self._summarize_documents(documents, summary_type)
    
    def summarize_multiple_files(self, file_paths: List[str], summary_type: str = "map_reduce") -> str:
        """Summarize multiple files."""
        documents = self.document_loader.load_multiple_files(file_paths)
        return self._summarize_documents(documents, summary_type)
    
    def _summarize_documents(self, documents: List[Document], summary_type: str) -> str:
        """Internal method to summarize documents using specified chain type."""
        if not documents:
            return "No content to summarize."
        
        # Choose the appropriate chain type based on document length
        if len(documents) == 1 and len(documents[0].page_content) < 4000:
            chain_type = "stuff"
        else:
            chain_type = summary_type
        
        chain = load_summarize_chain(
            self.llm, 
            chain_type=chain_type,
            verbose=False
        )
        
        try:
            summary = chain.run(documents)
            return summary.strip()
        except Exception as e:
            return f"Error during summarization: {str(e)}"
    
    def custom_summarize(self, text: str, custom_prompt: str) -> str:
        """Summarize with a custom prompt."""
        prompt_template = PromptTemplate(
            input_variables=["text"],
            template=custom_prompt + "\n\nText to summarize:\n{text}"
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        
        try:
            result = chain.run(text=text)
            return result.strip()
        except Exception as e:
            return f"Error during custom summarization: {str(e)}"
    
    def get_summary_stats(self, original_text: str, summary: str) -> Dict[str, Any]:
        """Get statistics about the summarization."""
        original_words = len(original_text.split())
        summary_words = len(summary.split())
        compression_ratio = summary_words / original_words if original_words > 0 else 0
        
        return {
            "original_words": original_words,
            "summary_words": summary_words,
            "compression_ratio": compression_ratio,
            "compression_percentage": f"{(1 - compression_ratio) * 100:.1f}%"
        }
