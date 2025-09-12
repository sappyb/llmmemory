# Evelyn AI - Student Simulation System

A sophisticated AI-powered system for simulating student interactions with different understanding levels, emotional states, and reasoning styles. This system is designed for educational research, teacher training, and curriculum development.

## 🎯 Features

### Student Persona Simulation
- **36 Different Student Personas** based on:
  - **Understanding Level**: High, Medium, Low
  - **Emotional State**: Engaged, Distressed, Fatigued, Anxious, Bored, Fed Up
  - **Reasoning Style**: Deductive, Analogical

### Two Interface Modes
1. **Web Interface** - Interactive Streamlit-based chat interface
2. **Socket Server** - Command-line server for programmatic access

### Advanced AI Capabilities
- **RAG (Retrieval Augmented Generation)** with FAISS vector database
- **Context-aware response generation** using OpenAI GPT models
- **Sentiment analysis** of user inputs
- **Multi-threaded server** for handling multiple clients

## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone <repository-url>
cd llmmemory

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
```bash
# Copy environment template
cp examples/student_simulator_env_example.txt .env

# Edit .env with your API keys
nano .env
```

### 3. Run the Application
```bash
# Start web interface
python main.py web

# Start socket server
python main.py server

# Start server on custom port
python main.py server --port 3000 --host 0.0.0.0
```

## 📁 Repository Structure

```
llmmemory/
├── main.py                          # Main entry point
├── app.py                           # Streamlit web interface
├── requirements.txt                 # Python dependencies
├── environment.yml                  # Conda environment
├── setup.py                        # Package setup
│
├── core/                           # Core modules
│   ├── config.py                   # Configuration management
│   ├── logger.py                   # Logging utilities
│   ├── rag_processor.py            # RAG processing
│   ├── socket_server.py            # Socket server
│   ├── student_simulator_config.py # Student simulator config
│   ├── student_persona_manager.py  # Persona management
│   ├── student_data_processor.py   # Data processing
│   ├── response_generator.py       # Response generation
│   └── student_simulator_server.py # Student simulator server
│
├── improved_versions/              # Improved implementations
│   ├── gen_improved.py            # Improved gen.py
│   └── gen_understanding_emotion_reasoning_improved.py
│
├── legacy_versions/                # Legacy implementations
│   ├── gen.py                     # Original gen.py
│   ├── gen-1.py, gen-2.py, etc.   # Various versions
│   └── ...
│
├── examples/                       # Example files
│   ├── test_client.py             # Test client for gen_improved
│   ├── test_student_simulator_client.py
│   └── student_simulator_env_example.txt
│
├── docs/                          # Documentation
│   ├── IMPROVEMENTS.md            # General improvements
│   └── STUDENT_SIMULATOR_IMPROVEMENTS.md
│
├── prompt_folders/                # Prompt templates
│   ├── prompts.py                 # Main prompts
│   ├── prompts.txt                # Text prompts
│   └── ...
│
├── train_docs/                    # Training documents
│   ├── *.pdf                      # PDF documents
│   └── *.csv                      # CSV data files
│
└── images/                        # Images and assets
    └── *.png                      # UI images
```

## 🎭 Student Persona System

### Persona Codes
The system uses 3-letter codes to represent student personas:

**Format**: `[Understanding][Emotion][Reasoning]`

- **Understanding**: H=High, M=Medium, L=Low
- **Emotion**: E=Engaged, D=Distressed, F=Fatigued, A=Anxious, B=Bored, FED=Fed Up
- **Reasoning**: D=Deductive, A=Analogical

### Example Personas
- `HED` - High understanding + Engaged + Deductive
- `LFA` - Low understanding + Fatigued + Analogical
- `MFED` - Medium understanding + Fed up + Deductive

### All 36 Personas
```
High Understanding:
  HED, HDD, HFD, HAD, HBD, HFED, HEA, HDA, HFA, HAA, HBA, HFEA

Medium Understanding:
  MED, MDD, MFD, MAD, MBD, MFED, MEA, MDA, MFA, MAA, MBA, MFEA

Low Understanding:
  LED, LDD, LFD, LAD, LBD, LFED, LEA, LDA, LFA, LAA, LBA, LFEA
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file with the following variables:

```bash
# Server Configuration
SERVER_HOST=127.0.0.1
SERVER_PORT=2004
MAX_CONNECTIONS=5

# Model Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4-turbo-preview
MAX_TOKENS=128
TEMPERATURE=0.7

# Data Configuration
DATA_PATH=./train_docs/Updated_Extracted_Data.csv
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=evelyn_ai.log
```

## 🖥️ Usage Examples

### Web Interface
```bash
python main.py web
```
- Open browser to `http://localhost:8501`
- Upload PDF documents
- Select student type and model
- Chat with simulated students

### Socket Server
```bash
python main.py server
```

#### Test Client
```bash
python examples/test_student_simulator_client.py
```

#### Manual Testing
```bash
# Connect to server
telnet 127.0.0.1 2004

# Send questions with persona codes
What is photosynthesis?::HED
How do plants grow?::LFA
Explain gravity::MFED
exit
```

## 📊 API Reference

### Socket Server Protocol
- **Input Format**: `question::PERSONA_CODE`
- **Output Format**: `response_sentiment_score`
- **Commands**: `exit` to disconnect

### Example Interaction
```
Input:  What is photosynthesis?::HED
Output: Photosynthesis is the process where plants use sunlight, water, and carbon dioxide to create glucose and oxygen. It's like a factory that converts light energy into chemical energy._0.8

Input:  How do plants grow?::LFA
Output: I think plants grow by... um, like, they just get bigger? Maybe they eat dirt or something? I'm not really sure how it works._-0.2
```

## 🛠️ Development

### Running Tests
```bash
# Test web interface
python main.py web

# Test socket server
python main.py server

# Test with client
python examples/test_student_simulator_client.py
```

### Adding New Personas
Edit `core/student_persona_manager.py` to add new persona combinations.

### Customizing Prompts
Edit `prompt_folders/prompts.py` to modify student response patterns.

## 📚 Documentation

- [General Improvements](docs/IMPROVEMENTS.md) - Overview of code improvements
- [Student Simulator Improvements](docs/STUDENT_SIMULATOR_IMPROVEMENTS.md) - Detailed improvements for student simulator

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for GPT models
- HuggingFace for embedding models
- LangChain for RAG framework
- Streamlit for web interface
- FAISS for vector search

## 📞 Support

For questions or issues, please:
1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed information

---

**Evelyn AI** - Simulating realistic student interactions for educational research and training.