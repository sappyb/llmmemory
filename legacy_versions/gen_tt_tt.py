import logging
import socket
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.chat import SystemMessagePromptTemplate
from questionmaker import NoOpLLMChain

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from pydantic import BaseModel

class SummarizerMixin(BaseModel):
    class Config:
        arbitrary_types_allowed = True


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Load and preprocess dataset
def load_data(data_path):
    try:
        data = pd.read_csv(data_path)
        data['Combined'] = (
            "Question: " + data['Question'] +
            " Understanding: " + data['Understanding'] +
            " Emotion: " + data['Emotion'] +
            " Reasoning: " + data['Reasoning'] +
            " Response: " + data['Response']
        )
        return data
    except Exception as e:
        logging.error(f"Failed to load or process data: {e}")
        raise


def build_faiss_index(data):
    try:
        langchain_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        metadata = [{"id": str(i)} for i in range(len(data))]
        faiss_index = FAISS.from_texts(
            texts=data['Combined'].tolist(),
            embedding=langchain_embeddings,
            metadatas=metadata
        )
        id_to_row = data.to_dict(orient='index')
        return faiss_index, langchain_embeddings.client, id_to_row
    except Exception as e:
        logging.error(f"Failed to build FAISS index: {e}")
        raise


def retrieve_response(user_input, faiss_index, id_to_row, embedding_model, selected_context):
    try:
        print(selected_context)
        user_embedding = embedding_model.encode([user_input])
        print(f"Shape of User Embedding: {user_embedding.shape}")
        results = faiss_index.similarity_search_with_score_by_vector(user_embedding[0], k=10)
        print(f"Similarity Search Results: {results}")
        for document, score in results:
            row_id = int(document.metadata.get('id', -1))
            if row_id == -1:
                continue
            response_row = id_to_row[row_id]
            if (
                response_row['Understanding'].strip() == selected_context['Understanding'] and
                response_row['Emotion'].strip() == selected_context['Emotion'] and
                response_row['Reasoning'].strip() == selected_context['Reasoning']
            ):
                logging.info(f"Matched response found. Score: {score}")
                return response_row['Response']
        logging.info("No matching response found based on context.")
        return "No response found based on context."
    except Exception as e:
        logging.error(f"Failed to retrieve response: {e}")
        return "No response found based on context."


# Select LLM Model and Fallback Prompt
llm_model = ChatOpenAI(model_name="gpt-4-turbo-preview")
fallback_prompt = """You are a middle school student who speaks colloquially.
    If the input is a question, give a false answer in under 15 words, showing that you have low understanding of the concept.
    If you don't know the answer, say you don't know and don't make up an answer.
    ----------------
    {context}
    Chat History:{chat_history}"""

# Initialize Conversation Chain
def get_conversation_chain(faiss_index, llm_model, fallback_prompt):
    memory = ConversationBufferMemory(memory_key='chat_history', output_key='answer', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm_model,
        chain_type="stuff",
        verbose="False",
        memory = memory,
        retriever=faiss_index.as_retriever(),
        return_source_documents = True)
    no_op_chain = NoOpLLMChain(llm=llm)
    conversation_chain.question_generator = no_op_chain
    # Set fallback prompt for unmatched queries
    system_message_prompt = SystemMessagePromptTemplate.from_template(fallback_prompt)
    conversation_chain.combine_docs_chain.llm_chain.prompt.messages[0] = system_message_prompt
    conversation_chain.combine_docs_chain.llm_chain.prompt.input_variables = ['context', 'question', 'chat_history']
    return conversation_chain


# Retrieve context based on student state
def get_student_context(student_code):
    context_map = {
        "ED": {"Understanding": "High understanding", "Emotion": "engaged", "Reasoning": "deductive"},
    }
    return context_map.get(student_code.upper(), None)


# Start the server
def start_server(data_path, host='127.0.0.1', port=2004):
    conn = None
    try:
        data = load_data(data_path)
        faiss_index, embedding_model, id_to_row = build_faiss_index(data)
        global_context = {"Understanding": "High understanding", "Emotion": "engaged", "Reasoning": "deductive"}
        selected_context = global_context
        conversation_chain = get_conversation_chain(faiss_index, llm_model, fallback_prompt)
        logging.info(f"Conversation Chain created")
        s = socket.socket()
        s.bind((host, port))
        s.listen(1)
        logging.info(f"Server started at {host}:{port}... Waiting for connections.")

        while True:
            conn, addr = s.accept()
            logging.info(f"Connection established with {addr}")
            user_question = conn.recv(100000).decode("utf-8").strip()

            if "::" in user_question:
                student_state = user_question.split("::")[-1].strip().upper()
                selected_context = get_student_context(student_state) or global_context
                user_question = user_question.split("::")[0].strip()
                print(f"Selected Context: {selected_context}")
                logging.info(f"Selected Context: {selected_context}")

            if user_question.lower() == "exit":
                logging.info("Exiting server.")
                conn.close()
                break

            if user_question:
                # Use conversation chain for response
                answer = conversation_chain.run({"query": user_question})
                sid_obj = SentimentIntensityAnalyzer()
                sentiment_score = sid_obj.polarity_scores(user_question)['compound']
                logging.info(f"Question: {user_question}, Response: {answer}, Sentiment: {sentiment_score:.2f}")
                response_final = f"{answer}_{sentiment_score:.2f}"
                conn.send(response_final.encode())
    except Exception as e:
        logging.error(f"Server error: {e}")
    finally:
        conn.close()
        logging.info("Server connection closed.")

if __name__ == '__main__':
    data_path = "./train_docs/Updated_Extracted_Data.csv"
    start_server(data_path)

