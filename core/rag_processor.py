"""
RAG (Retrieval Augmented Generation) processing module
"""
import os
from typing import List, Optional
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceHub
from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts.chat import SystemMessagePromptTemplate

from questionmaker import NoOpLLMChain
from prompts import (
    low_understanding_engaged_student_prompt,
    medium_understanding_engaged_student_prompt,
    zero_shot_high_understanding__student_prompt,
    few_shot_reasoning_low_understanding_student_prompt,
    few_shot_reasoning_medium_understanding_student_prompt,
    high_understanding_fed_up_student_prompt
)
from .logger import get_logger

logger = get_logger(__name__)

class RAGProcessor:
    """Handles RAG operations for document processing and conversation management"""
    
    def __init__(self, config):
        self.config = config
        self.vectorstore = None
        self.conversation = None
        
    def get_pdf_text(self, pdf_paths: List[str]) -> str:
        """Extract text from PDF files"""
        try:
            text = ""
            for pdf_path in pdf_paths:
                if not os.path.exists(pdf_path):
                    logger.warning(f"PDF file not found: {pdf_path}")
                    continue
                    
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text()
            logger.info(f"Extracted text from {len(pdf_paths)} PDF files")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDFs: {e}")
            raise
    
    def get_text_chunks(self, text: str) -> List[str]:
        """Split text into chunks for processing"""
        try:
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=self.config.document.chunk_size,
                chunk_overlap=self.config.document.chunk_overlap,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
            logger.info(f"Created {len(chunks)} text chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error creating text chunks: {e}")
            raise
    
    def get_vectorstore(self, text_chunks: List[str]) -> FAISS:
        """Create vector store from text chunks"""
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=self.config.document.embedding_model,
                model_kwargs={'device': "cpu"}
            )
            vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
            logger.info("Vector store created successfully")
            return vectorstore
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
    
    def get_llm(self, model_name: str):
        """Initialize the appropriate LLM based on model name"""
        try:
            if model_name == 'OpenAI GPT 3.5':
                return ChatOpenAI()
            elif model_name == 'gpt-4-turbo-preview':
                return ChatOpenAI(model_name="gpt-4-turbo-preview")
            elif model_name == 'Google flan-t5-xxl':
                return HuggingFaceHub(
                    repo_id="google/flan-t5-xxl", 
                    model_kwargs={"temperature": 0.5, "max_length": 512}
                )
            elif model_name == 'Mistral':
                if not self.config.model.mistral_api_token:
                    raise ValueError("Mistral API token not provided")
                return HuggingFaceEndpoint(
                    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                    max_length=128,
                    temperature=0.5,
                    token=self.config.model.mistral_api_token
                )
            else:
                raise ValueError(f"Unsupported model: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing LLM {model_name}: {e}")
            raise
    
    def get_student_prompt(self, student_type: str) -> str:
        """Get the appropriate prompt template for student type"""
        prompt_map = {
            'General': low_understanding_engaged_student_prompt(),
            'Engaged': medium_understanding_engaged_student_prompt(),
            'zero shot high': zero_shot_high_understanding__student_prompt(),
            'few shot low': few_shot_reasoning_low_understanding_student_prompt(),
            'few shot medium': few_shot_reasoning_medium_understanding_student_prompt(),
            'Fedup_H': high_understanding_fed_up_student_prompt()
        }
        
        if student_type not in prompt_map:
            logger.warning(f"Unknown student type: {student_type}, using default")
            student_type = 'General'
            
        return prompt_map[student_type]
    
    def get_conversation_chain(self, vectorstore: FAISS, model_name: str, student_type: str = 'Engaged Low'):
        """Create conversation chain with specified parameters"""
        try:
            # Initialize LLM
            llm = self.get_llm(model_name)
            
            # Create memory
            memory = ConversationBufferMemory(
                memory_key='chat_history', 
                output_key='answer', 
                return_messages=True
            )
            
            # Create conversation chain
            conv_rqa = ConversationalRetrievalChain.from_llm(
                llm=llm,
                chain_type="stuff",
                verbose=False,
                memory=memory,
                retriever=vectorstore.as_retriever(),
                return_source_documents=True
            )
            
            # Replace question generator with no-op
            no_op_chain = NoOpLLMChain(llm=llm)
            conv_rqa.question_generator = no_op_chain
            
            # Set student prompt
            modified_template = self.get_student_prompt(student_type)
            system_message_prompt = SystemMessagePromptTemplate.from_template(modified_template)
            
            if hasattr(conv_rqa.combine_docs_chain.llm_chain.prompt, 'messages'):
                conv_rqa.combine_docs_chain.llm_chain.prompt.messages[0] = system_message_prompt
            else:
                logger.warning("The 'messages' attribute does not exist in the prompt object")
            
            # Add chat_history as input variable
            conv_rqa.combine_docs_chain.llm_chain.prompt.input_variables = [
                'context', 'question', 'chat_history'
            ]
            
            logger.info(f"Conversation chain created for student type: {student_type}")
            return conv_rqa
            
        except Exception as e:
            logger.error(f"Error creating conversation chain: {e}")
            raise
    
    def initialize_from_pdf(self, pdf_paths: List[str], model_name: str, student_type: str):
        """Initialize the RAG system from PDF files"""
        try:
            # Extract text from PDFs
            raw_text = self.get_pdf_text(pdf_paths)
            if not raw_text.strip():
                raise ValueError("No text extracted from PDF files")
            
            # Create text chunks
            text_chunks = self.get_text_chunks(raw_text)
            
            # Create vector store
            self.vectorstore = self.get_vectorstore(text_chunks)
            
            # Create conversation chain
            self.conversation = self.get_conversation_chain(
                self.vectorstore, model_name, student_type
            )
            
            logger.info("RAG system initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}")
            raise
    
    def process_question(self, question: str) -> dict:
        """Process a question and return response"""
        try:
            if not self.conversation:
                raise ValueError("RAG system not initialized")
            
            response = self.conversation({'question': question})
            logger.info(f"Processed question: {question[:50]}...")
            return response
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            raise
