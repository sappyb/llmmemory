"""
Configuration management for Evelyn AI Socket Server
"""
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ServerConfig:
    """Server configuration settings"""
    host: str = "127.0.0.1"
    port: int = 2004
    max_connections: int = 5
    buffer_size: int = 100000
    timeout: int = 30

@dataclass
class ModelConfig:
    """Model configuration settings"""
    default_model: str = "gpt-4-turbo-preview"
    mistral_api_token: Optional[str] = None
    openai_api_key: Optional[str] = None
    huggingface_api_token: Optional[str] = None

@dataclass
class DocumentConfig:
    """Document processing configuration"""
    default_pdf_path: str = "./train_docs/BaselineSyntheticData_October10_2023.pdf"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

@dataclass
class AppConfig:
    """Main application configuration"""
    server: ServerConfig
    model: ModelConfig
    document: DocumentConfig
    log_level: str = "INFO"
    log_file: str = "evelyn_ai.log"

def load_config() -> AppConfig:
    """Load configuration from environment variables and defaults"""
    return AppConfig(
        server=ServerConfig(
            host=os.getenv("SERVER_HOST", "127.0.0.1"),
            port=int(os.getenv("SERVER_PORT", "2004")),
            max_connections=int(os.getenv("MAX_CONNECTIONS", "5")),
            buffer_size=int(os.getenv("BUFFER_SIZE", "100000")),
            timeout=int(os.getenv("TIMEOUT", "30"))
        ),
        model=ModelConfig(
            default_model=os.getenv("DEFAULT_MODEL", "gpt-4-turbo-preview"),
            mistral_api_token=os.getenv("MISTRAL_API_TOKEN"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            huggingface_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
        ),
        document=DocumentConfig(
            default_pdf_path=os.getenv("DEFAULT_PDF_PATH", "./train_docs/BaselineSyntheticData_October10_2023.pdf"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        ),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_file=os.getenv("LOG_FILE", "evelyn_ai.log")
    )
