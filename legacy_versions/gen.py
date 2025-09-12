from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
import sys
import os
import socket

from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
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
from prompts import low_understanding_engaged_student_prompt, high_understanding_engaged_student_prompt, low_understanding_engaged_student_prompt, medium_understanding_engaged_student_prompt, low_understanding_analogy_student_prompt, high_understanding_fed_up_student_prompt, zero_shot_high_understanding__student_prompt, few_shot_reasoning_medium_understanding_student_prompt, few_shot_reasoning_low_understanding_student_prompt
import os
#from langchain_ollama import ChatOllama
import tiktoken
from langchain_community.llms import HuggingFaceEndpoint

from pydantic import BaseModel

class SummarizerMixin(BaseModel):
    class Config:
        arbitrary_types_allowed = True

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
    elif model == 'gpt-4-turbo-preview': 
        llm = ChatOpenAI(model_name="gpt-4-turbo-preview")
    elif model == 'Google flan-t5-xxl':
        llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    elif model == 'Facebook LLAMA':
        pass
        #llm = ChatOllama(model = "llama3")
        #llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin",model_type="llama",config={'max_new_tokens':128,'temperature':0.01})
    elif model == 'Mistral':
        HUGGINGFACEHUB_API_TOKEN = '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'
        repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
        llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.5, token=HUGGINGFACEHUB_API_TOKEN)
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
        modified_template = low_understanding_engaged_student_prompt()
    elif student_type == 'Engaged':
        modified_template = low_understanding_analogy_student_prompt()
    elif student_type == 'zero shot high':
        modified_template = zero_shot_high_understanding__student_prompt()
    elif student_type == 'few shot low':
        modified_template = few_shot_reasoning_low_understanding_student_prompt()
    elif student_type == 'few shot medium':
        modified_template = few_shot_reasoning_medium_understanding_student_prompt()
    elif student_type == 'Fedup_H':
        modified_template = high_understanding_fed_up_student_prompt()
    system_message_prompt = SystemMessagePromptTemplate.from_template(modified_template)
    if hasattr(conv_rqa.combine_docs_chain.llm_chain.prompt, 'messages'):
        conv_rqa.combine_docs_chain.llm_chain.prompt.messages[0] = system_message_prompt
    else:
        print("The 'messages' attribute does not exist in the prompt object.")

    # add chat_history as a variable to the llm_chain's ChatPromptTemplate object
    conv_rqa.combine_docs_chain.llm_chain.prompt.input_variables = ['context', 'question', 'chat_history']
  
    return conv_rqa

def select_model(model='OpenAI'):
    if model == 'OpenAI':
       model = 'gpt-4-turbo-preview'
       #model = 'OpenAI GPT 3.5'
       #model = 'Facebook LLAMA'
    else:
        model = 'Mistral'
    return model


def handle_userinput(user_question):
    response = conversation({'question': user_question})
    chat_history = response['chat_history']

def main():
    student_type = input("""Enter student type: 

            1. General(Low understanding engaged ), 
            2. Engaged(Analogy student), 
            3. Few shot medium (In development), 
            4. Few shot low (In Development),
            5. Fedup_H (High understanding Fedup student),
            6. Zero shot high (In development)
            
            :: """)
    load_dotenv()

    #select model
    model = select_model(model=str(input('''Enter :
        1. OpenAI
        2. Mistral
        :: ''')))

    #select student type
    pdf_docs = ["./train_docs/BaselineSyntheticData_October10_2023.pdf"]
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
                        vectorstore, model, student_type)
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
        if ":" in user_question:
            print(user_question)
            student_state = user_question.split(":")[-1].strip()
            user_question = user_question.split(":")[0].strip()
            print(user_question)
            print("user_question is printed")
        if user_question == "Exit":
            conn.close()
        if user_question:
             response = conversation({'question': user_question})
             chat_history = response['chat_history']
             sid_obj = SentimentIntensityAnalyzer()
             sentiment_question = sid_obj.polarity_scores(user_question)['compound']
             sentiment_mean = (sentiment_question)
             print("Question : {}".format(user_question))
             print("Response : {}".format(response['answer']))
             voice_response = "{}==={}".format("Student_1", response['answer'])
             response_final = "{}_{}".format(response['answer'], sentiment_mean)
             conn.send(response_final.encode())
    conn.close()
if __name__ == '__main__':
    main()
