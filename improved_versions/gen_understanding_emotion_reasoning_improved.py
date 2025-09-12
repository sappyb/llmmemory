#!/usr/bin/env python3
"""
Improved Student Understanding Emotion Reasoning Simulator
Enhanced version with better architecture, security, and error handling
"""
import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from student_simulator_config import load_config
from logger import setup_logger
from student_simulator_server import StudentSimulatorServer

class StudentSimulatorApp:
    """Main application class for Student Understanding Emotion Reasoning Simulator"""
    
    def __init__(self):
        self.config = load_config()
        self.logger = setup_logger(
            name="student_simulator",
            log_level=self.config.logging.level,
            log_file=self.config.logging.file
        )
        self.server = None
        
    def validate_config(self) -> bool:
        """Validate configuration and required files"""
        try:
            # Check if data file exists
            if not Path(self.config.data.data_path).exists():
                self.logger.error(f"Data file not found: {self.config.data.data_path}")
                return False
            
            # Check OpenAI API key
            if not self.config.model.openai_api_key:
                self.logger.error("OpenAI API key not provided")
                return False
            
            self.logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    def display_persona_codes(self) -> None:
        """Display available persona codes"""
        from student_persona_manager import StudentPersonaManager
        
        persona_manager = StudentPersonaManager()
        personas = persona_manager.list_personas()
        
        print("\n" + "="*80)
        print("AVAILABLE STUDENT PERSONA CODES")
        print("="*80)
        print("Format: [Understanding][Emotion][Reasoning]")
        print("\nUnderstanding: H=High, M=Medium, L=Low")
        print("Emotion: E=Engaged, D=Distressed, F=Fatigued, A=Anxious, B=Bored, FED=Fed Up")
        print("Reasoning: D=Deductive, A=Analogical")
        print("\n" + "-"*80)
        
        # Group by understanding level
        for understanding in ["High", "Medium", "Low"]:
            print(f"\n{understanding} Understanding:")
            understanding_personas = [p for p in personas if understanding in p.understanding]
            
            for persona in understanding_personas:
                emotion_short = persona.emotion[0].upper() if persona.emotion != "fed up" else "FED"
                print(f"  {persona.code:6} - {persona.emotion:10} + {persona.reasoning:10} reasoning")
        
        print("\n" + "="*80)
        print("USAGE: Send 'question::PERSONA_CODE' to the server")
        print("EXAMPLE: 'What is photosynthesis?::LED'")
        print("="*80)
    
    def run(self):
        """Main application entry point"""
        try:
            self.logger.info("Starting Student Understanding Emotion Reasoning Simulator")
            
            # Validate configuration
            if not self.validate_config():
                self.logger.error("Configuration validation failed")
                return 1
            
            # Display persona codes
            self.display_persona_codes()
            
            # Create and initialize server
            self.server = StudentSimulatorServer(self.config)
            
            if not self.server.initialize():
                self.logger.error("Failed to initialize server")
                return 1
            
            # Display server status
            status = self.server.get_server_status()
            print(f"\nServer Status:")
            print(f"  Host: {status['host']}")
            print(f"  Port: {status['port']}")
            print(f"  Data Records: {status['data_records']}")
            print(f"  Available Personas: {status['available_personas']}")
            print(f"\nServer starting... Press Ctrl+C to stop")
            print("-" * 50)
            
            # Start server
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
    app = StudentSimulatorApp()
    sys.exit(app.run())

if __name__ == '__main__':
    main()
