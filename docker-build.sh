#!/bin/bash
# Docker build script for Evelyn AI

set -e

echo "🐳 Building Evelyn AI Docker Image"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
IMAGE_NAME="evelyn-ai"
TAG="latest"
BUILD_ARGS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        -n|--name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --no-cache)
            BUILD_ARGS="--no-cache"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -t, --tag TAG        Set image tag (default: latest)"
            echo "  -n, --name NAME      Set image name (default: evelyn-ai)"
            echo "  --no-cache          Build without cache"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"

echo -e "${BLUE}Image: ${FULL_IMAGE_NAME}${NC}"
echo -e "${BLUE}Build args: ${BUILD_ARGS}${NC}"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Warning: .env file not found. Creating from template...${NC}"
    if [ -f "examples/student_simulator_env_example.txt" ]; then
        cp examples/student_simulator_env_example.txt .env
        echo -e "${YELLOW}Please edit .env file with your API keys before running the container.${NC}"
    else
        echo -e "${RED}Error: No .env template found. Please create .env file manually.${NC}"
        exit 1
    fi
fi

# Build the Docker image
echo -e "${BLUE}Building Docker image...${NC}"
if docker build ${BUILD_ARGS} -t "${FULL_IMAGE_NAME}" .; then
    echo -e "${GREEN}✅ Docker image built successfully: ${FULL_IMAGE_NAME}${NC}"
else
    echo -e "${RED}❌ Docker build failed${NC}"
    exit 1
fi

# Show image information
echo ""
echo -e "${BLUE}Image Information:${NC}"
docker images "${FULL_IMAGE_NAME}"

echo ""
echo -e "${GREEN}🎉 Build completed successfully!${NC}"
echo ""
echo "Next steps:"
echo "1. Run web interface: docker run -p 8501:8501 ${FULL_IMAGE_NAME}"
echo "2. Run socket server: docker run -p 2004:2004 ${FULL_IMAGE_NAME} python main.py server"
echo "3. Use docker-compose: docker-compose up"
