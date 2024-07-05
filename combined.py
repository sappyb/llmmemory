import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, RetrievalQA, ConversationChain
from langchain import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub, OpenAI
from langchain.memory.vectorstore import VectorStoreRetrieverMemory
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain import PromptTemplate
#from langchain.llms import CTransformers
from langchain.callbacks import get_openai_callback
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import SystemMessagePromptTemplate
from htmlTempletes import css, bot_template, user_template
from questionmaker import NoOpLLMChain
from prompts import general_prompt, engaged_student_prompt, engagedlow_student_prompt, engagedchild_student_prompt
import os
import tiktoken
import time
import numpy as np
import openai
import faiss

# Global Variables declarations

# Initialize FAISS Index for Vectors for Domain Specific Knowledge
dimension = 1536  # Dimension of the embeddings
index = faiss.IndexFlatL2(dimension)
metadata_pd = []

# Initialize FAISS Index for Vectors for Interaction History
dimension = 1536  # Dimension of the embeddings
index_recall = faiss.IndexFlatL2(dimension)
metadata = []

# Define Functions to convert PDFs to on huge section of text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Define Functions to chunkize text
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Define Functions to vectorize the chunks of text
def vectorize_text(text):
    response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
    return np.array(response['data'][0]['embedding'])


# Define Functions to store the PDFs in the FAISS Index
def store_pdf(pdf_docs):
    text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(text)
    
    #Vectorize text chunks
    for chunk in text_chunks:
        timestamp = time.time()
        vector = vectorize_text(chunk)
        
        '''
        metadata.append({
            "content": chunk,
            "timestamp": vector
        })
        '''
        index.add(np.array([vector]))

# Non dependent functions
        
# Check if a text is a question
def is_question(text):
    question_words = {"what", "why", "how", "is", "are", "was", "were", "do", "does", "did", "can", "could", "should", "would", "which", "who", "whom", "whose", "where", "when"}
    words = text.lower().split()
    if words and words[0] in question_words:
        return True
    return re.search(r'\?\s*$', text.strip()) is not None

# Define Functions to calculate cosine similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Define Functions to calculate relevance
def calculate_relevance(memory_vec, input_vec):
    return cosine_similarity(memory_vec, input_vec)

# Define Functions to calculate exponential decay
def exponential_decay(relevance, elapsed_time, decay_rate):
    return relevance * np.exp(-decay_rate * elapsed_time)

# Define Functions to calculate recall probability
def recall_probability(relevance, elapsed_time, decay_rate):
    return 1 - np.exp(-exponential_decay(relevance, elapsed_time, decay_rate))

# Summarize text to reduce its length
def summarize_text(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Please summarize the following text:\n\n{text}"}
        ],
        max_tokens=150  # Adjust as needed
    )
    return response.choices[0].message['content']

# Truncate text to fit within token limit
def truncate_text(text, max_tokens):
    tokens = text.split()
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return " ".join(tokens)

# Define Functions to recall memory from the database
def recall_memory(user_id, input_vec, threshold=0.10, decay_rate=0.001):
    current_time = time.time()
    if len(metadata) == 0:
        return None
    distances, indices = index_recall.search(np.array([input_vec]), len(metadata))
    for idx in indices[0]:
        print(idx)
        if idx == -1:
            continue
        memory_vec = index_recall.reconstruct(int(idx))
        relevance = calculate_relevance(memory_vec, input_vec)
        elapsed_time = current_time - metadata[idx]["timestamp"]
        prob = recall_probability(relevance, elapsed_time, decay_rate)
        if prob > threshold:
            return metadata[idx]["content"]
    return None

# Memory Storage
def store_memory(user_id, content):
    if is_question(content):
        return
    vector = vectorize_text(content)
    timestamp = time.time()
    metadata.append({
        "user_id": user_id,
        "content": content,
        "timestamp": timestamp,
    })
    index_recall.add(np.array([vector]))

#########################################################################################

# Streamlit App selection of Model    
def select_model():
    model = st.selectbox(
    'Select the model you want to use',
    ('OpenAI GPT 3.5', 'Google flan-t5-xxl', 'Facebook LLAMA 7b'))
    return model

# Streamlit App selection of Student Type
def select_student_type():
    student_type = st.selectbox(
    'Select the type of student you want to be',
    ('General', 'Engaged', 'Engaged Low', 'Engaged Child'))
    return student_type


def get_conversation_chain(model, student_type):
    model = select_model()
    temperature = st.slider('Select the temperature', 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    if model == 'OpenAI GPT 3.5': 
        llm = 'gpt-3.5-turbo-0125'
    elif model == 'OpenAI GPT 4o':
        llm = 'gpt-4o'
    else:
        st.error('Model name not valid', icon="🚨")
    llm = OpenAI(model_name=llm, temperature=0.7, api_key=openai.api_key)
    use_chat_history = 0
    if use_chat_history == 1:
        memory = ConversationBufferMemory(memory_key='chat_history', output_key='answer', return_messages=True)
    else:
        memory = ConversationBufferMemory()
    chain = ConversationChain(llm=llm, memory=memory)

    no_op_chain = NoOpLLMChain(llm=llm)
    chain.question_generator = no_op_chain
    if student_type == 'General':
        modified_template = general_prompt()
    elif student_type == 'Engaged':
        modified_template = engaged_student_prompt()
    elif student_type == 'Engaged Low':
        modified_template = engagedlow_student_prompt()
    elif student_type == 'Engaged Child':
        modified_template = engagedchild_student_prompt()
    
    system_message_prompt = SystemMessagePromptTemplate.from_template(modified_template)
    chain.prompt.messages[0] = system_message_prompt
    
    # Add chat_history as a variable to the llm_chain's ChatPromptTemplate object
    chain.prompt.input_variables = ['context', 'question', 'chat_history']
  
    return chain

def get_interaction_input():
    input_vec = vectorize_text(input_text)
    recalled_memory = recall_memory(user_id, input_vec)
    print("Recalled Memory:", recalled_memory)
    if recalled_memory != None:
      combined_input = recalled_memory
    else:
      combined_input = ''
    # Calculate the available tokens for input_text and truncate accordingly
    max_total_tokens = 4097
    reserved_completion_tokens = 256
    max_input_tokens = max_total_tokens - reserved_completion_tokens

    combined_input = truncate_text(combined_input, max_input_tokens)

def get_response():
    student = select_student_type()
    model = select_model()
    conversation_chain = get_conversation_chain(model=model, student_type=student)
    if conversation_chain:
        combined_input = get_interaction_input()
        response = conversation_chain.run(input=combined_input)
        print(response)
