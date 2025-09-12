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
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain import PromptTemplate
#from langchain.llms import CTransformers
from langchain.callbacks import get_openai_callback
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import SystemMessagePromptTemplate
from htmlTempletes import css, bot_template, user_template
from questionmaker import NoOpLLMChain
from prompts import general_prompt, general_engaged_deductive_student_prompt, general_bored_deductive_student_prompt, general_anxious_deductive_student_prompt, general_distressed_deductive_student_prompt, general_fatigued_deductive_student_prompt, general_fedup_deductive_student_prompt, general_engaged_analogy_student_prompt, general_bored_analogy_student_prompt, general_anxious_analogy_student_prompt, general_distressed_analogy_student_prompt, general_fatigued_analogy_student_prompt, general_fedup_analogy_student_prompt
import os
#from langchain_ollama import ChatOllama
import tiktoken
from langchain_community.llms import HuggingFaceEndpoint

global_conversation = None
global_student_type = None
global_chat_history = None

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
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device':"cpu"})
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore



def get_system_message_prompt(student_type):

    """Get the appropriate system message prompt based on student type."""
    if student_type == 'General':
        return SystemMessagePromptTemplate.from_template(general_prompt())

    elif student_type == 'Engaged_Deductive':
        return SystemMessagePromptTemplate.from_template(general_engaged_deductive_student_prompt())

    elif student_type == 'Bored_Deductive':
        return SystemMessagePromptTemplate.from_template(general_bored_deductive_student_prompt())

    elif student_type == 'Anxious_Deductive':
        return SystemMessagePromptTemplate.from_template(general_anxious_deductive_student_prompt())

    elif student_type == 'Distressed_Deductive':
        return SystemMessagePromptTemplate.from_template(general_distressed_deductive_student_prompt())

    elif student_type == 'Fatigued_Deductive':
        return SystemMessagePromptTemplate.from_template(general_fatigued_deductive_student_prompt())

    elif student_type == 'Fedup_Deductive':
        return SystemMessagePromptTemplate.from_template(general_fedup_deductive_student_prompt())

    elif student_type == 'Engaged_Analogy':
        return SystemMessagePromptTemplate.from_template(general_engaged_analogy_student_prompt())

    elif student_type == 'Bored_Analogy':
        return SystemMessagePromptTemplate.from_template(general_bored_analogy_student_prompt())

    elif student_type == 'Anxious_Analogy':
        return SystemMessagePromptTemplate.from_template(general_anxious_analogy_student_prompt())

    elif student_type == 'Distressed_Analogy':
        return SystemMessagePromptTemplate.from_template(general_distressed_analogy_student_prompt())

    elif student_type == 'Fatigued_Analogy':
        return SystemMessagePromptTemplate.from_template(general_fatigued_analogy_student_prompt())

    elif student_type == 'Fedup_Analogy':
        return SystemMessagePromptTemplate.from_template(general_fedup_analogy_student_prompt())

    else:
        return SystemMessagePromptTemplate.from_template(general_prompt())  # Default to general prompt




def get_conversation_chain(vectorstore, model, student_type='Engaged Low'):
    #create llm
    if model == 'OpenAI GPT 3.5': 
        llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    elif model == 'gpt-4-turbo-preview': 
        llm = ChatOpenAI(model="gpt-4o")
    elif model == 'Google flan-t5-xxl':
        llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    elif model == 'Facebook LLAMA':
        pass
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
                                                 memory=memory,
                                                 retriever=vectorstore.as_retriever(),
                                                 return_source_documents = True)

    no_op_chain = NoOpLLMChain(llm=llm)
    conv_rqa.question_generator = no_op_chain
    # Update the system prompt for the first run

    conv_rqa.combine_docs_chain.llm_chain.prompt = ChatPromptTemplate.from_messages([get_system_message_prompt(student_type)])


    # add chat_history as a variable to the llm_chain's ChatPromptTemplate object
    conv_rqa.combine_docs_chain.llm_chain.prompt.input_variables = ['context', 'question', 'chat_history']
  
    return conv_rqa

def select_model(model='OpenAI'):
    if model == 'OpenAI':
       model = 'gpt-4-turbo-preview'
    else:
        model = 'Mistral'
    return model

def handle_userinput(user_question):
    global global_student_type
    global global_conversation
    global global_chat_history
    student_type = ''
    if "::" in user_question:
        if user_question.split("::")[-1].strip().upper() == 'ED':
            student_type = 'Engaged_Deductive'
        elif user_question.split("::")[-1].strip().upper() == 'BD':
            student_type = 'Bored_Deductive'
        elif user_question.split("::")[-1].strip().upper() == 'AD':
            student_type = 'Anxious_Deductive'
        elif user_question.split("::")[-1].strip().upper() == 'DD':
            student_type = 'Distressed_Deductive'
        elif user_question.split("::")[-1].strip().upper() == 'FD':
            student_type = 'Fatigued_Deductive'
        elif user_question.split("::")[-1].strip().upper() == 'FED':
            student_type = 'Fedup_Deductive'
        elif user_question.split("::")[-1].strip().upper() == 'EA':
            student_type = 'Engaged_Analogy'
        elif user_question.split("::")[-1].strip().upper() == 'BA':
            student_type = 'Bored_Analogy'
        elif user_question.split("::")[-1].strip().upper() == 'AA':
            student_type = 'Anxious_Analogy'
        elif user_question.split("::")[-1].strip().upper() == 'DA':
            student_type = 'Distressed_Analogy'
        elif user_question.split("::")[-1].strip().upper() == 'FA':
            student_type = 'Fatigued_Analogy'
        elif user_question.split("::")[-1].strip().upper() == 'FEA':
            student_type = 'Fedup_Analogy'
        else:
            student_type = global_student_type
    else:
        student_type = global_student_type
    print(student_type)
    system_prompt = get_system_message_prompt(student_type)
    #global_conversation.combine_docs_chain.llm_chain.prompt.messages[0] = system_prompt
    global_conversation.combine_docs_chain.llm_chain.prompt = ChatPromptTemplate.from_messages([get_system_message_prompt(student_type)])
    cleaned_user_question = user_question.split("::")[0].strip()
    response = global_conversation({'question': cleaned_user_question})
    global_chat_history = response['chat_history']
    return response

def main():
    student_type_number = int(input("""Enter student type: 

            1. 'General' -> Low understanding engaged student,
            2. 'Engaged_Deductive',
            3. 'Bored_Deductive',
            4. 'Anxious_Deductive',
            5. 'Distressed_Deductive',
            6. 'Fatigued_Deductive',
            7. 'Fedup_Deductive',
            8. 'Engaged_Analogy',
            9. 'Bored_Analogy',
            10. 'Anxious_Analogy',
            11. 'Distressed_Analogy',
            12. 'Fatigued_Analogy',
            13. 'Fedup_Analogy'
            
            :: """))
    student_type = ''
    if student_type_number == 1:
        student_type = 'General'
    elif student_type_number == 2:
        student_type = 'Engaged_Deductive'
    elif student_type_number == 3:
        student_type = 'Bored_Deductive'
    elif student_type_number == 4:
        student_type = 'Anxious_Deductive'
    elif student_type_number == 5:
        student_type = 'Distressed_Deductive'
    elif student_type_number == 6:
        student_type = 'Fatigued_Deductive'
    elif student_type_number == 7:
        student_type = 'Fedup_Deductive'
    elif student_type_number == 8:
        student_type = 'Engaged_Analogy'
    elif student_type_number == 9:
        student_type = 'Bored_Analogy'
    elif student_type_number == 10:
        student_type = 'Anxious_Analogy'
    elif student_type_number == 11:
        student_type = 'Distressed_Analogy'
    elif student_type_number == 12:
        student_type = 'Fatigued_Analogy'
    elif student_type_number == 12:
        student_type = 'Fedup_Analogy'
    load_dotenv()

    #select model
    model = select_model(model=str(input('''Enter :
        1. OpenAI
        2. Mistral
        :: ''')))


    global global_conversation
    global global_chat_history
    global global_student_type

    global_student_type = student_type

    #select student type
    pdf_docs = ["./train_docs/Baseline.pdf"]
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
        global_conversation = get_conversation_chain(
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
        if user_question == "Exit":
            conn.close()
        if user_question:
             response = handle_userinput(user_question)
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
