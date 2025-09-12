#!/usr/bin/env python3
"""
Improved Evelyn AI Socket Server
Enhanced version of gen.py with better architecture, security, and error handling
"""
import sys
import os
import signal
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import load_config
from logger import setup_logger
from rag_processor import RAGProcessor
from socket_server import EvelynAIServer

class EvelynAIApp:
    """Main application class for Evelyn AI Socket Server"""
    
    def __init__(self):
        self.config = load_config()
        self.logger = setup_logger(
            name="evelyn_ai",
            log_level=self.config.log_level,
            log_file=self.config.log_file
        )
        self.server = None
        self.rag_processor = RAGProcessor(self.config)
        
    def validate_config(self) -> bool:
        """Validate configuration and required files"""
        try:
            # Check if default PDF exists
            if not Path(self.config.document.default_pdf_path).exists():
                self.logger.error(f"Default PDF not found: {self.config.document.default_pdf_path}")
                return False
            
            # Check API tokens for selected models
            if not self.config.model.openai_api_key:
                self.logger.warning("OpenAI API key not set - OpenAI models will not work")
            
            if not self.config.model.mistral_api_token:
                self.logger.warning("Mistral API token not set - Mistral model will not work")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    def get_user_input(self) -> tuple[str, str]:
        """Get user input for model and student type selection"""
        print("\n" + "="*50)
        print("Evelyn AI Socket Server - Configuration")
        print("="*50)
        
        # Student type selection
        print("\nSelect student type:")
        student_types = {
            "1": ("General", "Low understanding engaged"),
            "2": ("Engaged", "Medium understanding engaged"),
            "3": ("few shot medium", "In development"),
            "4": ("few shot low", "In development"),
            "5": ("Fedup_H", "High understanding fed-up student"),
            "6": ("zero shot high", "In development")
        }
        
        for key, (name, desc) in student_types.items():
            print(f"  {key}. {name} ({desc})")
        
        while True:
            choice = input("\nEnter choice (1-6): ").strip()
            if choice in student_types:
                student_type = student_types[choice][0]
                break
            print("Invalid choice. Please enter 1-6.")
        
        # Model selection
        print("\nSelect model:")
        models = {
            "1": ("OpenAI", "GPT-4 Turbo Preview (requires API key)"),
            "2": ("Mistral", "Mistral-7B-Instruct (requires API token)")
        }
        
        for key, (name, desc) in models.items():
            print(f"  {key}. {name} ({desc})")
        
        while True:
            choice = input("\nEnter choice (1-2): ").strip()
            if choice in models:
                model_name = models[choice][0]
                break
            print("Invalid choice. Please enter 1-2.")
        
        # Map to actual model names
        model_mapping = {
            "OpenAI": "gpt-4-turbo-preview",
            "Mistral": "Mistral"
        }
        
        return model_mapping[model_name], student_type
    
    def initialize_rag_system(self, model_name: str, student_type: str) -> bool:
        """Initialize the RAG system"""
        try:
            self.logger.info(f"Initializing RAG system with model: {model_name}, student: {student_type}")
            
            # Initialize RAG processor
            self.rag_processor.initialize_from_pdf(
                pdf_paths=[self.config.document.default_pdf_path],
                model_name=model_name,
                student_type=student_type
            )
            
            self.logger.info("RAG system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG system: {e}")
            return False
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down gracefully...")
            if self.server:
                self.server.stop_server()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def run(self):
        """Main application entry point"""
        try:
            self.logger.info("Starting Evelyn AI Socket Server")
            
            # Validate configuration
            if not self.validate_config():
                self.logger.error("Configuration validation failed")
                return 1
            
            # Get user input
            model_name, student_type = self.get_user_input()
            
            # Initialize RAG system
            if not self.initialize_rag_system(model_name, student_type):
                self.logger.error("Failed to initialize RAG system")
                return 1
            
            # Setup signal handlers
            self.setup_signal_handlers()
            
            # Create and start server
            self.server = EvelynAIServer(self.config)
            self.server.rag_processor = self.rag_processor
            
            print(f"\nServer starting on {self.config.server.host}:{self.config.server.port}")
            print("Press Ctrl+C to stop the server")
            print("="*50)
            
            self.server.start_server()
            
        except KeyboardInterrupt:
            self.logger.info("Server stopped by user")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return 1
        finally:
            if self.server:
                self.server.stop_server()
        
        return 0

def main():
    """Main entry point"""
    app = EvelynAIApp()
    sys.exit(app.run())

if __name__ == '__main__':
    main()
