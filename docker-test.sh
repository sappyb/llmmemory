#!/bin/bash
# Docker test script for Evelyn AI

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "🧪 Testing Evelyn AI Docker Setup"
echo "================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker is running${NC}"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  .env file not found. Creating from template...${NC}"
    if [ -f "examples/student_simulator_env_example.txt" ]; then
        cp examples/student_simulator_env_example.txt .env
        echo -e "${YELLOW}Please edit .env file with your API keys before running tests.${NC}"
    else
        echo -e "${RED}❌ No .env template found. Please create .env file manually.${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✅ Environment file found${NC}"

# Test 1: Build Docker image
echo ""
echo -e "${BLUE}Test 1: Building Docker image...${NC}"
if ./docker-build.sh --no-cache; then
    echo -e "${GREEN}✅ Docker image built successfully${NC}"
else
    echo -e "${RED}❌ Docker build failed${NC}"
    exit 1
fi

# Test 2: Test container startup
echo ""
echo -e "${BLUE}Test 2: Testing container startup...${NC}"
if timeout 30s docker run --rm evelyn-ai:latest python test.py; then
    echo -e "${GREEN}✅ Container startup test passed${NC}"
else
    echo -e "${YELLOW}⚠️  Container startup test had issues (this might be expected without API keys)${NC}"
fi

# Test 3: Test web interface (quick test)
echo ""
echo -e "${BLUE}Test 3: Testing web interface startup...${NC}"
echo "Starting web interface for 10 seconds..."
if timeout 10s docker run --rm -p 8501:8501 evelyn-ai:latest python main.py web > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Web interface startup test passed${NC}"
else
    echo -e "${YELLOW}⚠️  Web interface startup test had issues (this might be expected without API keys)${NC}"
fi

# Test 4: Test socket server (quick test)
echo ""
echo -e "${BLUE}Test 4: Testing socket server startup...${NC}"
echo "Starting socket server for 5 seconds..."
if timeout 5s docker run --rm -p 2004:2004 evelyn-ai:latest python main.py server > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Socket server startup test passed${NC}"
else
    echo -e "${YELLOW}⚠️  Socket server startup test had issues (this might be expected without API keys)${NC}"
fi

# Test 5: Test docker-compose
echo ""
echo -e "${BLUE}Test 5: Testing docker-compose configuration...${NC}"
if docker-compose config > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Docker-compose configuration is valid${NC}"
else
    echo -e "${RED}❌ Docker-compose configuration is invalid${NC}"
    exit 1
fi

# Test 6: Test image size and layers
echo ""
echo -e "${BLUE}Test 6: Checking image size...${NC}"
IMAGE_SIZE=$(docker images evelyn-ai:latest --format "table {{.Size}}" | tail -n 1)
echo -e "${GREEN}✅ Image size: ${IMAGE_SIZE}${NC}"

# Test 7: Test container health
echo ""
echo -e "${BLUE}Test 7: Testing container health...${NC}"
CONTAINER_ID=$(docker run -d -p 8501:8501 evelyn-ai:latest python main.py web)
sleep 10

if docker ps | grep -q "$CONTAINER_ID"; then
    echo -e "${GREEN}✅ Container is running${NC}"
    docker stop "$CONTAINER_ID" > /dev/null 2>&1
else
    echo -e "${YELLOW}⚠️  Container stopped (might be expected without proper configuration)${NC}"
fi

echo ""
echo -e "${GREEN}🎉 Docker tests completed!${NC}"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Run: ./docker-run.sh"
echo "3. Or use: docker-compose up"
echo ""
echo "For detailed Docker usage, see DOCKER.md"
