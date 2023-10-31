from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
import sys
import os
import socket

from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub
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
from prompts import low_understanding_engaged_student_prompt, medium_understanding_engaged_student_prompt, low_understanding_bored_student_prompt, high_understanding_fed_up_student_prompt
import os
import tiktoken

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    #embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device':"cpu"})
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore, model, student_type='Engaged Low'):
    #create llm
    if model == 'OpenAI GPT 3.5': 
        llm = ChatOpenAI()
    elif model == 'Google flan-t5-xxl':
        llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    elif model == 'Facebook LLAMA':
        pass
        #llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin",model_type="llama",config={'max_new_tokens':128,'temperature':0.01})
    else:
        print('Model name not valid')
    #create memory type
    memory = ConversationBufferMemory(memory_key='chat_history', output_key='answer', return_messages=True)
    #create conversation chain
    conv_rqa = ConversationalRetrievalChain.from_llm(llm=llm,
                                                 chain_type="stuff",
                                                 verbose="False",
                                                 memory = memory,
                                                 retriever=vectorstore.as_retriever(),
                                                 return_source_documents = True)

    no_op_chain = NoOpLLMChain(llm=llm)
    conv_rqa.question_generator = no_op_chain
    if student_type == 'General':
        modified_template = low_understanding_bored_student_prompt()
    elif student_type == 'Engaged':
        modified_template = medium_understanding_engaged_student_prompt()
    elif student_type == 'Engaged Low':
        modified_template = low_understanding_engaged_student_prompt()
    elif student_type == 'Engaged Child':
        modified_template = high_understanding_fed_up_student_prompt()
    system_message_prompt = SystemMessagePromptTemplate.from_template(modified_template)
    conv_rqa.combine_docs_chain.llm_chain.prompt.messages[0] = system_message_prompt

    # add chat_history as a variable to the llm_chain's ChatPromptTemplate object
    conv_rqa.combine_docs_chain.llm_chain.prompt.input_variables = ['context', 'question', 'chat_history']
  
    return conv_rqa

def select_model():
    model = 'OpenAI GPT 3.5'
    return model


def handle_userinput(user_question):
    response = conversation({'question': user_question})
    chat_history = response['chat_history']

def main():
    load_dotenv()

    #select model
    model = select_model()

    #select student type
    pdf_docs = ["/home/isltmile/evelyn6_train.pdf"]
    # get pdf text
    raw_text = get_pdf_text(pdf_docs)  
    if raw_text == "":
         print("Please upload at least one PDF")
    else:
        # get the text chunks
        text_chunks = get_text_chunks(raw_text)
        # create vector store
        vectorstore = get_vectorstore(text_chunks)
        # create conversation chain
        conversation = get_conversation_chain(
                        vectorstore, model)
    host = '127.0.0.1'
    port = int(2004)
    s = socket.socket()
    s.bind((host, port))
    s.listen(1)
    print("Start")
    while True:
        conn, addr = s.accept()
        user_question = conn.recv(100000)
        user_question = user_question.decode("utf-8")
        if user_question == "Exit":
            conn.close()
        if user_question:
             response = conversation({'question': user_question})
             chat_history = response['chat_history']
             sid_obj = SentimentIntensityAnalyzer()
             sentiment_question = sid_obj.polarity_scores(user_question)['compound']
             sentiment_mean = (sentiment_question)
             response_final = "{}_{}".format(response['answer'], sentiment_mean)
             conn.send(response_final.encode())
    conn.close()
if __name__ == '__main__':
    main()
