#!/bin/bash
# Docker run script for Evelyn AI

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
IMAGE_NAME="evelyn-ai"
TAG="latest"
MODE="web"
PORT="8501"
HOST="0.0.0.0"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -h|--host)
            HOST="$2"
            shift 2
            ;;
        -i|--image)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -m, --mode MODE      Run mode: web or server (default: web)"
            echo "  -p, --port PORT      Port to expose (default: 8501 for web, 2004 for server)"
            echo "  -h, --host HOST      Host to bind to (default: 0.0.0.0)"
            echo "  -i, --image IMAGE    Docker image name (default: evelyn-ai)"
            echo "  -t, --tag TAG        Docker image tag (default: latest)"
            echo "  --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run web interface on port 8501"
            echo "  $0 -m server -p 2004                 # Run socket server on port 2004"
            echo "  $0 -m web -p 8080                    # Run web interface on port 8080"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"

# Set default port based on mode
if [ "$MODE" = "server" ] && [ "$PORT" = "8501" ]; then
    PORT="2004"
fi

echo "🐳 Running Evelyn AI Docker Container"
echo "===================================="
echo -e "${BLUE}Mode: ${MODE}${NC}"
echo -e "${BLUE}Image: ${FULL_IMAGE_NAME}${NC}"
echo -e "${BLUE}Port: ${PORT}${NC}"
echo -e "${BLUE}Host: ${HOST}${NC}"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

# Check if image exists
if ! docker images "${FULL_IMAGE_NAME}" | grep -q "${TAG}"; then
    echo -e "${YELLOW}Image ${FULL_IMAGE_NAME} not found. Building it first...${NC}"
    ./docker-build.sh -n "${IMAGE_NAME}" -t "${TAG}"
fi

# Create necessary directories
mkdir -p logs data

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Warning: .env file not found. Creating from template...${NC}"
    if [ -f "examples/student_simulator_env_example.txt" ]; then
        cp examples/student_simulator_env_example.txt .env
        echo -e "${YELLOW}Please edit .env file with your API keys before running the container.${NC}"
        echo -e "${YELLOW}Press Enter to continue or Ctrl+C to edit .env file first...${NC}"
        read
    else
        echo -e "${RED}Error: No .env template found. Please create .env file manually.${NC}"
        exit 1
    fi
fi

# Run the container
echo -e "${BLUE}Starting container...${NC}"

if [ "$MODE" = "web" ]; then
    echo -e "${GREEN}🌐 Starting web interface on http://${HOST}:${PORT}${NC}"
    docker run -it --rm \
        -p "${HOST}:${PORT}:8501" \
        -v "$(pwd)/logs:/app/logs" \
        -v "$(pwd)/train_docs:/app/train_docs" \
        -v "$(pwd)/.env:/app/.env" \
        "${FULL_IMAGE_NAME}" \
        python improved_versions/gen_understanding_emotion_reasoning_improved.py
elif [ "$MODE" = "server" ]; then
    echo -e "${GREEN}🔌 Starting socket server on ${HOST}:${PORT}${NC}"
    docker run -it --rm \
        -p "${HOST}:${PORT}:2004" \
        -v "$(pwd)/logs:/app/logs" \
        -v "$(pwd)/train_docs:/app/train_docs" \
        -v "$(pwd)/.env:/app/.env" \
        "${FULL_IMAGE_NAME}" \
        python improved_versions/gen_understanding_emotion_reasoning_improved.py
else
    echo -e "${RED}Error: Invalid mode '${MODE}'. Use 'web' or 'server'.${NC}"
    exit 1
fi
