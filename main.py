#!/usr/bin/env python3
"""
Evelyn AI - Student Simulation System
Main entry point for the Evelyn AI student simulation system

This application provides two main modes:
1. Web Interface (Streamlit) - Interactive web-based chat interface
2. Socket Server - Command-line socket server for programmatic access

Usage:
    python main.py web          # Start web interface
    python main.py server       # Start socket server
    python main.py --help       # Show help
"""
import sys
import argparse
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Evelyn AI - Student Simulation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py web                    # Start web interface
    python main.py server                 # Start socket server
    python main.py server --port 3000    # Start server on custom port
        """
    )
    
    parser.add_argument(
        'mode',
        choices=['web', 'server'],
        help='Mode to run: web (Streamlit interface) or server (socket server)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=2004,
        help='Port for socket server (default: 2004)'
    )
    
    parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Host for socket server (default: 127.0.0.1)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'web':
        start_web_interface()
    elif args.mode == 'server':
        start_socket_server(args.host, args.port)

def start_web_interface():
    """Start the Streamlit web interface"""
    try:
        import streamlit.web.cli as stcli
        import os
        
        # Set the app file
        app_file = str(Path(__file__).parent / "app.py")
        
        # Start Streamlit
        sys.argv = ["streamlit", "run", app_file]
        stcli.main()
        
    except ImportError:
        print("Error: Streamlit not installed. Please install it with: pip install streamlit")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting web interface: {e}")
        sys.exit(1)

def start_socket_server(host, port):
    """Start the socket server"""
    try:
        from core import load_student_config, setup_logger, StudentSimulatorServer
        
        # Set environment variables for host and port
        import os
        os.environ['SERVER_HOST'] = host
        os.environ['SERVER_PORT'] = str(port)
        
        # Load configuration and setup logging
        config = load_student_config()
        logger = setup_logger(
            name="student_simulator",
            log_level=config.logging.level,
            log_file=config.logging.file
        )
        
        # Create and start server
        server = StudentSimulatorServer(config)
        
        if not server.initialize():
            logger.error("Failed to initialize server")
            sys.exit(1)
        
        logger.info(f"Starting server on {host}:{port}")
        server.start_server()
        
    except ImportError as e:
        print(f"Error: Required modules not found: {e}")
        print("Please ensure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting socket server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
