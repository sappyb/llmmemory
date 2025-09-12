#!/usr/bin/env python3
"""
Test client for Evelyn AI Socket Server
Demonstrates how to connect and interact with the improved server
"""
import socket
import sys
import time

def test_client(host='127.0.0.1', port=2004):
    """Test client for the Evelyn AI Socket Server"""
    
    print(f"Connecting to Evelyn AI Server at {host}:{port}")
    
    try:
        # Create socket connection
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((host, port))
        
        print("Connected successfully!")
        print("Type 'exit' to quit, or 'question:student_state' for specific student state")
        print("="*50)
        
        while True:
            # Get user input
            user_input = input("\nEnter your question: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() == 'exit':
                client_socket.send(user_input.encode())
                break
            
            # Send question to server
            client_socket.send(user_input.encode())
            
            # Receive response
            response = client_socket.recv(100000).decode('utf-8')
            
            # Parse response (format: "answer_sentiment_score")
            if '_' in response:
                answer, sentiment = response.rsplit('_', 1)
                print(f"\nAnswer: {answer}")
                print(f"Sentiment Score: {sentiment}")
            else:
                print(f"\nResponse: {response}")
        
    except ConnectionRefusedError:
        print("Error: Could not connect to server. Make sure the server is running.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client_socket.close()
        print("\nDisconnected from server.")

if __name__ == '__main__':
    # Allow custom host and port
    host = sys.argv[1] if len(sys.argv) > 1 else '127.0.0.1'
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 2004
    
    test_client(host, port)
