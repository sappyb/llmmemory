# Docker Containerization Guide

This guide explains how to run Evelyn AI using Docker containers for easy deployment and development.

## 🐳 Docker Setup

### Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+
- Git

### Quick Start

#### 1. Clone and Navigate
```bash
git clone <repository-url>
cd llmmemory
git checkout docker-containerization
```

#### 2. Configure Environment
```bash
# Copy environment template
cp examples/student_simulator_env_example.txt .env

# Edit with your API keys
nano .env
```

#### 3. Build and Run
```bash
# Build Docker image
./docker-build.sh

# Run web interface
./docker-run.sh

# Run socket server
./docker-run.sh -m server -p 2004
```

## 📁 Docker Files

### Core Files
- `Dockerfile` - Main container definition
- `docker-compose.yml` - Multi-service orchestration
- `docker-compose.override.yml` - Development overrides
- `.dockerignore` - Files to exclude from build

### Scripts
- `docker-build.sh` - Build Docker image
- `docker-run.sh` - Run containers with options

## 🚀 Usage Options

### Option 1: Docker Compose (Recommended)

#### Start All Services
```bash
# Start web interface and socket server
docker-compose up

# Start in background
docker-compose up -d

# View logs
docker-compose logs -f
```

#### Start Specific Service
```bash
# Web interface only
docker-compose up evelyn-web

# Socket server only
docker-compose up evelyn-server

# Run tests
docker-compose --profile test up evelyn-test
```

#### Stop Services
```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Option 2: Docker Commands

#### Build Image
```bash
# Basic build
./docker-build.sh

# Custom tag
./docker-build.sh -t v1.0.0

# No cache build
./docker-build.sh --no-cache
```

#### Run Container
```bash
# Web interface (default)
./docker-run.sh

# Socket server
./docker-run.sh -m server -p 2004

# Custom port
./docker-run.sh -m web -p 8080

# Custom image
./docker-run.sh -i my-evelyn -t v1.0.0
```

#### Manual Docker Commands
```bash
# Build
docker build -t evelyn-ai .

# Run web interface
docker run -p 8501:8501 evelyn-ai

# Run socket server
docker run -p 2004:2004 evelyn-ai python main.py server

# Run with volumes
docker run -p 8501:8501 \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/train_docs:/app/train_docs \
  -v $(pwd)/.env:/app/.env \
  evelyn-ai
```

## 🔧 Configuration

### Environment Variables
Create `.env` file with:
```bash
# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=2004

# Model Configuration
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4-turbo-preview

# Logging
LOG_LEVEL=INFO
LOG_FILE=/app/logs/evelyn_ai.log
```

### Volume Mounts
- `./logs:/app/logs` - Log files
- `./train_docs:/app/train_docs` - Training data
- `./.env:/app/.env` - Environment configuration

### Ports
- `8501` - Web interface (Streamlit)
- `2004` - Socket server

## 🛠️ Development

### Development Mode
```bash
# Use override file for development
docker-compose -f docker-compose.yml -f docker-compose.override.yml up

# Mount source code for live development
docker-compose up evelyn-web
```

### Debugging
```bash
# Run with debug logging
docker run -e LOG_LEVEL=DEBUG -p 8501:8501 evelyn-ai

# Interactive shell
docker run -it evelyn-ai /bin/bash

# Check logs
docker logs <container_id>
```

### Testing
```bash
# Run tests
docker-compose --profile test up evelyn-test

# Manual test
docker run evelyn-ai python test.py
```

## 📊 Monitoring

### Health Checks
- Web interface: `http://localhost:8501/healthz`
- Socket server: Connection test on port 2004

### Logs
```bash
# View all logs
docker-compose logs

# Follow logs
docker-compose logs -f

# View specific service logs
docker-compose logs evelyn-web
```

### Container Status
```bash
# List running containers
docker-compose ps

# Container stats
docker stats

# Inspect container
docker inspect <container_name>
```

## 🚀 Production Deployment

### Production Configuration
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  evelyn-web:
    restart: always
    environment:
      - LOG_LEVEL=WARNING
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
```

### Security Considerations
- Use non-root user (already configured)
- Mount volumes as read-only where possible
- Use secrets management for API keys
- Enable container scanning

### Scaling
```bash
# Scale web interface
docker-compose up --scale evelyn-web=3

# Use load balancer
# Configure nginx/traefik for load balancing
```

## 🔍 Troubleshooting

### Common Issues

#### Container Won't Start
```bash
# Check logs
docker-compose logs evelyn-web

# Check environment
docker-compose config

# Rebuild without cache
docker-compose build --no-cache
```

#### Port Already in Use
```bash
# Check port usage
netstat -tulpn | grep :8501

# Use different port
./docker-run.sh -p 8080
```

#### Permission Issues
```bash
# Fix log directory permissions
sudo chown -R $USER:$USER logs/

# Rebuild with proper permissions
docker-compose build --no-cache
```

#### API Key Issues
```bash
# Check .env file
cat .env

# Verify environment in container
docker exec <container> env | grep OPENAI
```

### Debug Commands
```bash
# Container shell
docker exec -it evelyn-ai-web /bin/bash

# Check Python path
docker exec evelyn-ai-web python -c "import sys; print(sys.path)"

# Test imports
docker exec evelyn-ai-web python -c "from core import load_config; print('OK')"
```

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Streamlit Docker Guide](https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker)
- [Python Docker Best Practices](https://pythonspeed.com/docker/)

## 🤝 Contributing

When adding Docker-related changes:
1. Test with both `docker-compose` and manual `docker` commands
2. Update this documentation
3. Ensure `.dockerignore` is up to date
4. Test on different platforms (Linux, macOS, Windows)

---

**Happy Containerizing!** 🐳
