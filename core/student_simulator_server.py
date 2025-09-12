"""
Enhanced Socket Server for Student Understanding Emotion Reasoning Simulator
"""
import socket
import threading
import time
import signal
import sys
from typing import Optional, Dict, Any, Tuple
from contextlib import contextmanager

from student_simulator_config import AppConfig
from student_persona_manager import StudentPersonaManager, StudentPersona
from student_data_processor import StudentDataProcessor
from response_generator import ResponseGenerator, ResponseContext
from logger import get_logger

logger = get_logger(__name__)

class StudentSimulatorServer:
    """Enhanced socket server for student simulation"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.server_socket: Optional[socket.socket] = None
        self.running = False
        self.active_connections: Dict[str, socket.socket] = {}
        
        # Initialize components
        self.persona_manager = StudentPersonaManager()
        self.data_processor = StudentDataProcessor(config)
        self.response_generator = ResponseGenerator(config)
        
        # Current context
        self.current_context = self.persona_manager.get_context("HED")  # Default
        
        # Setup signal handlers
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            self.stop_server()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def initialize(self) -> bool:
        """Initialize the server and load data"""
        try:
            logger.info("Initializing Student Simulator Server...")
            
            # Load and process data
            self.data_processor.load_data()
            self.data_processor.build_faiss_index()
            
            logger.info("Server initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize server: {e}")
            return False
    
    def start_server(self) -> None:
        """Start the socket server"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.config.server.host, self.config.server.port))
            self.server_socket.listen(self.config.server.max_connections)
            self.server_socket.settimeout(1.0)  # Non-blocking with timeout
            
            self.running = True
            logger.info(f"Server started on {self.config.server.host}:{self.config.server.port}")
            logger.info(f"Available persona codes: {', '.join(self.persona_manager.get_available_codes()[:10])}...")
            
            while self.running:
                try:
                    conn, addr = self.server_socket.accept()
                    conn.settimeout(self.config.server.timeout)
                    
                    # Handle connection in a separate thread
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(conn, addr),
                        daemon=True
                    )
                    client_thread.start()
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        logger.error(f"Error accepting connection: {e}")
                        
        except Exception as e:
            logger.error(f"Error starting server: {e}")
            raise
        finally:
            self.stop_server()
    
    def stop_server(self) -> None:
        """Stop the socket server"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        logger.info("Server stopped")
    
    def _handle_client(self, conn: socket.socket, addr: Tuple[str, int]) -> None:
        """Handle individual client connection"""
        client_id = f"{addr[0]}:{addr[1]}"
        self.active_connections[client_id] = conn
        
        try:
            logger.info(f"Client connected: {client_id}")
            
            while self.running:
                try:
                    # Receive data from client
                    data = conn.recv(self.config.server.buffer_size)
                    if not data:
                        break
                    
                    user_input = data.decode("utf-8").strip()
                    logger.info(f"Received from {client_id}: {user_input}")
                    
                    # Handle exit command
                    if user_input.lower() == "exit":
                        logger.info(f"Client {client_id} requested exit")
                        break
                    
                    # Process the input
                    response = self._process_input(user_input)
                    
                    # Send response back to client
                    conn.send(response.encode())
                    logger.info(f"Sent response to {client_id}")
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    logger.error(f"Error handling client {client_id}: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Error in client handler for {client_id}: {e}")
        finally:
            # Clean up connection
            if client_id in self.active_connections:
                del self.active_connections[client_id]
            conn.close()
            logger.info(f"Client disconnected: {client_id}")
    
    def _process_input(self, user_input: str) -> str:
        """Process user input and generate response"""
        try:
            # Parse input for persona code
            question, persona_code = self._parse_input(user_input)
            
            # Get persona and context
            persona = self.persona_manager.get_persona(persona_code)
            context = self.persona_manager.get_context(persona_code)
            
            # Search for similar responses
            similar_records = self.data_processor.search_similar(question)
            
            # Find matching response
            response = self._find_matching_response(question, similar_records, context, persona)
            
            # Analyze sentiment
            sentiment_score = self.response_generator.analyze_sentiment(question)
            
            # Format and return response
            return self.response_generator.format_response(response, sentiment_score)
            
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            return f"Error processing request: {str(e)}_0.0"
    
    def _parse_input(self, user_input: str) -> Tuple[str, str]:
        """Parse user input to extract question and persona code"""
        if "::" in user_input:
            parts = user_input.split("::", 1)
            question = parts[0].strip()
            persona_code = parts[1].strip().upper()
        else:
            question = user_input
            persona_code = "HED"  # Default persona
        
        return question, persona_code
    
    def _find_matching_response(self, question: str, similar_records: list, context: Dict[str, str], persona: StudentPersona) -> str:
        """Find matching response based on context"""
        try:
            # Look for exact context match
            for record_idx, distance in similar_records:
                record = self.data_processor.get_record_by_index(record_idx)
                if record is None:
                    continue
                
                if self.data_processor.validate_context_match(record, context):
                    # Generate refined response
                    response_context = ResponseContext(
                        question=question,
                        raw_answer=record['Response'],
                        understanding=context['Understanding'],
                        emotion=context['Emotion'],
                        reasoning=context['Reasoning'],
                        persona_code=persona.code
                    )
                    
                    refined_response = self.response_generator.generate_refined_response(response_context)
                    
                    if self.response_generator.validate_response(refined_response):
                        logger.info(f"Found matching response for persona {persona.code}")
                        return refined_response
            
            # No matching response found, generate fallback
            logger.info(f"No matching response found, generating fallback for persona {persona.code}")
            return self.response_generator.generate_fallback_response(question, persona)
            
        except Exception as e:
            logger.error(f"Error finding matching response: {e}")
            return "I don't know how to answer that."
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get current server status"""
        return {
            "running": self.running,
            "active_connections": len(self.active_connections),
            "host": self.config.server.host,
            "port": self.config.server.port,
            "data_records": self.data_processor.get_statistics().get("total_records", 0),
            "available_personas": len(self.persona_manager.get_available_codes())
        }
    
    @contextmanager
    def connection_manager(self, conn: socket.socket):
        """Context manager for socket connections"""
        try:
            yield conn
        finally:
            conn.close()
