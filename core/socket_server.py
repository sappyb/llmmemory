"""
Enhanced Socket Server for Evelyn AI
"""
import socket
import threading
import time
from typing import Optional, Dict, Any
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import AppConfig
from rag_processor import RAGProcessor
from logger import get_logger

logger = get_logger(__name__)

class EvelynAIServer:
    """Enhanced socket server for Evelyn AI chatbot"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.server_socket: Optional[socket.socket] = None
        self.running = False
        self.rag_processor = RAGProcessor(config)
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.active_connections: Dict[str, socket.socket] = {}
        
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
            
            while self.running:
                try:
                    conn, addr = self.server_socket.accept()
                    conn.settimeout(self.config.server.timeout)
                    
                    # Handle connection in a separate thread
                    client_thread = threading.Thread(
                        target=self.handle_client,
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
    
    def handle_client(self, conn: socket.socket, addr: tuple) -> None:
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
                    
                    user_question = data.decode("utf-8").strip()
                    logger.info(f"Received from {client_id}: {user_question}")
                    
                    # Handle exit command
                    if user_question.lower() == "exit":
                        logger.info(f"Client {client_id} requested exit")
                        break
                    
                    # Process the question
                    response = self.process_question(user_question)
                    
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
    
    def process_question(self, user_question: str) -> str:
        """Process a question and return formatted response"""
        try:
            # Parse question and student state if provided
            student_state = None
            if ":" in user_question:
                parts = user_question.split(":", 1)
                user_question = parts[0].strip()
                student_state = parts[1].strip()
                logger.info(f"Student state: {student_state}")
            
            # Get response from RAG processor
            response = self.rag_processor.process_question(user_question)
            answer = response['answer']
            
            # Analyze sentiment
            sentiment_score = self.sentiment_analyzer.polarity_scores(user_question)['compound']
            
            # Format response
            response_final = f"{answer}_{sentiment_score}"
            
            logger.info(f"Question: {user_question}")
            logger.info(f"Response: {answer}")
            logger.info(f"Sentiment: {sentiment_score}")
            
            return response_final
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return f"Error processing question: {str(e)}_0.0"
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get current server status"""
        return {
            "running": self.running,
            "active_connections": len(self.active_connections),
            "host": self.config.server.host,
            "port": self.config.server.port
        }
