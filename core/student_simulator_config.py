"""
Configuration management for Student Understanding Emotion Reasoning Simulator
"""
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
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
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4-turbo-preview"
    max_tokens: int = 128
    temperature: float = 0.7
    embedding_model: str = "all-MiniLM-L6-v2"

@dataclass
class DataConfig:
    """Data processing configuration"""
    data_path: str = "./train_docs/Updated_Extracted_Data.csv"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_retrieval_results: int = 117

@dataclass
class StudentConfig:
    """Student persona configuration"""
    default_context: Dict[str, str] = None
    
    def __post_init__(self):
        if self.default_context is None:
            self.default_context = {
                "Understanding": "High understanding",
                "Emotion": "engaged", 
                "Reasoning": "deductive"
            }

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    file: str = "student_simulator.log"

@dataclass
class AppConfig:
    """Main application configuration"""
    server: ServerConfig
    model: ModelConfig
    data: DataConfig
    student: StudentConfig
    logging: LoggingConfig

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
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"),
            max_tokens=int(os.getenv("MAX_TOKENS", "128")),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        ),
        data=DataConfig(
            data_path=os.getenv("DATA_PATH", "./train_docs/Updated_Extracted_Data.csv"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            max_retrieval_results=int(os.getenv("MAX_RETRIEVAL_RESULTS", "117"))
        ),
        student=StudentConfig(),
        logging=LoggingConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            format=os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"),
            file=os.getenv("LOG_FILE", "student_simulator.log")
        )
    )
