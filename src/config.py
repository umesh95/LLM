"""Configuration settings for the text summarization project."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the summarization project."""
    
    # API Keys
    PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Model settings
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama-3.1-sonet-large-128k-online")
    PERPLEXITY_BASE_URL = "https://api.perplexity.ai"
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4000"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
    
    # Chunking settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Supported file types
    SUPPORTED_EXTENSIONS = ['.txt', '.pdf', '.docx', '.md']
    
    @classmethod
    def validate_config(cls):
        """Validate that required configuration is present."""
        if not cls.PERPLEXITY_API_KEY:
            raise ValueError("PERPLEXITY_API_KEY is required. Please set it in your .env file.")
        return True
