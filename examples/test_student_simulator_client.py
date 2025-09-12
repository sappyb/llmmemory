#!/usr/bin/env python3
"""
Test client for Student Understanding Emotion Reasoning Simulator
Demonstrates how to interact with different student personas
"""
import socket
import sys
import time

def test_student_simulator(host='127.0.0.1', port=2004):
    """Test client for the Student Simulator Server"""
    
    print("="*80)
    print("STUDENT UNDERSTANDING EMOTION REASONING SIMULATOR - TEST CLIENT")
    print("="*80)
    print("Available Persona Codes:")
    print("  HED - High understanding + Engaged + Deductive")
    print("  LED - Low understanding + Engaged + Deductive") 
    print("  MFA - Medium understanding + Fatigued + Analogical")
    print("  HFED - High understanding + Fed up + Deductive")
    print("  LBA - Low understanding + Bored + Analogical")
    print("  And many more...")
    print("\nFormat: question::PERSONA_CODE")
    print("Example: What is photosynthesis?::LED")
    print("="*80)
    
    try:
        # Create socket connection
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((host, port))
        
        print(f"Connected to Student Simulator Server at {host}:{port}")
        print("Type 'exit' to quit")
        print("-" * 50)
        
        # Test with some example questions
        test_questions = [
            ("What is photosynthesis?", "HED"),
            ("How do plants make food?", "LED"), 
            ("What happens in the water cycle?", "MFA"),
            ("Explain gravity to me", "HFED"),
            ("What is DNA?", "LBA")
        ]
        
        print("\nRunning test questions...")
        for question, persona in test_questions:
            print(f"\nTesting: {question} with persona {persona}")
            test_input = f"{question}::{persona}"
            client_socket.send(test_input.encode())
            response = client_socket.recv(100000).decode('utf-8')
            
            if '_' in response:
                answer, sentiment = response.rsplit('_', 1)
                print(f"Answer: {answer}")
                print(f"Sentiment: {sentiment}")
            else:
                print(f"Response: {response}")
            
            time.sleep(1)  # Brief pause between questions
        
        print("\n" + "="*50)
        print("Interactive mode - Enter your own questions:")
        print("Format: question::PERSONA_CODE")
        print("="*50)
        
        while True:
            # Get user input
            user_input = input("\nEnter question and persona (or 'exit'): ").strip()
            
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
        print("Run: python gen_understanding_emotion_reasoning_improved.py")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client_socket.close()
        print("\nDisconnected from server.")

if __name__ == '__main__':
    # Allow custom host and port
    host = sys.argv[1] if len(sys.argv) > 1 else '127.0.0.1'
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 2004
    
    test_student_simulator(host, port)
