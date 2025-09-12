import os
import socket
import logging
import openai
import pandas as pd
import faiss
from langchain.docstore import InMemoryDocstore
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.chat import SystemMessagePromptTemplate
from pydantic import BaseModel



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Custom Pydantic Model Config for LangChain Compatibility
class Config(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True
    }

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

# Generate embeddings and build FAISS index
def build_faiss_index(data):
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(data['Combined'].tolist())
        dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(embeddings)
        id_to_row = {i: row for i, row in data.iterrows()}
        # Create a SimpleDocstore and index-to-docstore mapping
        docstore = InMemoryDocstore({str(i): row.to_dict() for i, row in data.iterrows()})
        index_to_docstore_id = {i: str(i) for i in range(len(data))}
        return faiss_index, model, id_to_row, docstore, index_to_docstore_id
    except Exception as e:
        logging.error(f"Failed to build FAISS index: {e}")
        raise

def refine_answer_with_openai(question, retrieved_answer, understanding):
    """
    Use OpenAI's GPT-3.5-turbo model to generate a refined answer.
    """
    if understanding == 'Low understanding':

       prompt = (
           f"User question: {question}\n"
           f"Context: {retrieved_answer}\n\n"
           "Make sure the emotion and understading is same as the context and generate a refined answer based on the information in the context."
       )
    elif understanding == 'Medium understanding':
       prompt = (
           f"User question: {question}\n"
           f"Context: {retrieved_answer}\n\n"
           "Make sure the emotion and understading is same as the context and generate a refined answer based on the information in the context."
       )
    else:
       prompt = (
           f"User question: {question}\n"
           f"Context: {retrieved_answer}\n\n"
           "Make sure the emotion and understading is same as the context and generate a refined answer based on the information in the context."
       )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=128,
        temperature=0.7
    )

    refined_answer = response.choices[0].message["content"]
    return refined_answer

# Retrieve Response Function
def retrieve_response(user_input, faiss_index, id_to_row, selected_context, model):
    try:
        user_embedding = model.encode([user_input])
        k = min(117, faiss_index.ntotal)  # Adjust dynamically
        _, indices = faiss_index.search(user_embedding, k=k)
        for row_id in indices[0]:
            response_row = id_to_row[row_id]
            if (
                response_row['Understanding'].strip() == selected_context['Understanding'] and
                response_row['Emotion'].strip() == selected_context['Emotion'] and
                response_row['Reasoning'].strip() == selected_context['Reasoning']
            ):
                logging.info(f"Matched response found based on context. Question : {user_input} Reponse : {response_row['Response']}")
                # Retrieve raw answer from the dataset
                raw_answer = response_row['Response']
                print(f'Raw Answer : {raw_answer}')
                # 4) Pass the raw answer + user’s question to the LLM for refinement
                refined_answer = refine_answer_with_openai(user_input, raw_answer, selected_context['Understanding'])
                print(f'Refined Answer : {refined_answer}')
                # Return the first refined answer that fits the context
                return refined_answer
        logging.info("No matching response found based on context.")
        return "Silent"  # No suitable match found
    except Exception as e:
        logging.error(f"Failed to retrieve response: {e}")
        return "No response found based on context."


# Conversation Chain with LangChain
def get_conversation_chain(faiss_index, llm_model, fallback_prompt):
    memory = ConversationBufferMemory(memory_key='chat_history', output_key='answer', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm_model,
        retriever=faiss_index,
        memory=memory,
        return_source_documents=False
    )

    # Set fallback prompt for unmatched queries
    system_message_prompt = SystemMessagePromptTemplate.from_template(fallback_prompt)
    conversation_chain.combine_docs_chain.llm_chain.prompt.messages[0] = system_message_prompt
    return conversation_chain

# Retrieve context based on student state
def get_student_context(student_code):
    context_map = {
        "HED": {"Understanding": "High understanding", "Emotion": "engaged", "Reasoning": "deductive"},
        "HDD": {"Understanding": "High understanding", "Emotion": "distressed", "Reasoning": "deductive"},
        "HFD": {"Understanding": "High understanding", "Emotion": "fatigued", "Reasoning": "deductive"},
        "HAD": {"Understanding": "High understanding", "Emotion": "anxious", "Reasoning": "deductive"},
        "HBD": {"Understanding": "High understanding", "Emotion": "bored", "Reasoning": "deductive"},
        "HFED": {"Understanding": "High understanding", "Emotion": "fed up", "Reasoning": "deductive"},
        "HEA": {"Understanding": "High understanding", "Emotion": "engaged", "Reasoning": "analogical"},
        "HDA": {"Understanding": "High understanding", "Emotion": "distressed", "Reasoning": "analogical"},
        "HFA": {"Understanding": "High understanding", "Emotion": "fatigued", "Reasoning": "analogical"},
        "HAA": {"Understanding": "High understanding", "Emotion": "anxious", "Reasoning": "analogical"},
        "HBA": {"Understanding": "High understanding", "Emotion": "bored", "Reasoning": "analogical"},
        "HFEA": {"Understanding": "High understanding", "Emotion": "fed up", "Reasoning": "analogical"},
        "MED": {"Understanding": "Medium understanding", "Emotion": "engaged", "Reasoning": "deductive"},
        "MDD": {"Understanding": "Medium understanding", "Emotion": "distressed", "Reasoning": "deductive"},
        "MFD": {"Understanding": "Medium understanding", "Emotion": "fatigued", "Reasoning": "deductive"},
        "MAD": {"Understanding": "Medium understanding", "Emotion": "anxious", "Reasoning": "deductive"},
        "MBD": {"Understanding": "Medium understanding", "Emotion": "bored", "Reasoning": "deductive"},
        "MFED": {"Understanding": "Medium understanding", "Emotion": "fed up", "Reasoning": "deductive"},
        "MEA": {"Understanding": "Medium understanding", "Emotion": "engaged", "Reasoning": "analogical"},
        "MDA": {"Understanding": "Medium understanding", "Emotion": "distressed", "Reasoning": "analogical"},
        "MFA": {"Understanding": "Medium understanding", "Emotion": "fatigued", "Reasoning": "analogical"},
        "MAA": {"Understanding": "Medium understanding", "Emotion": "anxious", "Reasoning": "analogical"},
        "MBA": {"Understanding": "Medium understanding", "Emotion": "bored", "Reasoning": "analogical"},
        "MFEA": {"Understanding": "Medium understanding", "Emotion": "fed up", "Reasoning": "analogical"},
        "LED": {"Understanding": "Low understanding", "Emotion": "engaged", "Reasoning": "deductive"},
        "LDD": {"Understanding": "Low understanding", "Emotion": "distressed", "Reasoning": "deductive"},
        "LFD": {"Understanding": "Low understanding", "Emotion": "fatigued", "Reasoning": "deductive"},
        "LAD": {"Understanding": "Low understanding", "Emotion": "anxious", "Reasoning": "deductive"},
        "LBD": {"Understanding": "Low understanding", "Emotion": "bored", "Reasoning": "deductive"},
        "LFED": {"Understanding": "Low understanding", "Emotion": "fed up", "Reasoning": "deductive"},
        "LEA": {"Understanding": "Low understanding", "Emotion": "engaged", "Reasoning": "analogical"},
        "LDA": {"Understanding": "Low understanding", "Emotion": "distressed", "Reasoning": "analogical"},
        "LFA": {"Understanding": "Low understanding", "Emotion": "fatigued", "Reasoning": "analogical"},
        "LAA": {"Understanding": "Low understanding", "Emotion": "anxious", "Reasoning": "analogical"},
        "LBA": {"Understanding": "Low understanding", "Emotion": "bored", "Reasoning": "analogical"},
        "LFEA": {"Understanding": "Low understanding", "Emotion": "fed up", "Reasoning": "analogical"},
    }
    return context_map.get(student_code.upper(), None)


# Main Function
def main():
    # Load and process data
    data_path = "./train_docs/Updated_Extracted_Data.csv"
    data = load_data(data_path)
    global_context = {"Understanding": "High understanding", "Emotion": "engaged", "Reasoning": "deductive"}
    selected_context = global_context
    # Build FAISS index
    faiss_index, model, id_to_row, docstore, index_to_docstore_id = build_faiss_index(data)
    faiss_retriever = FAISS(index=faiss_index, embedding_function=SentenceTransformer('all-MiniLM-L6-v2'), docstore=docstore, index_to_docstore_id=index_to_docstore_id)
    retriever = faiss_retriever.as_retriever()


    # Select LLM Model and Fallback Prompt
    llm_model = ChatOpenAI(model_name="gpt-4-turbo-preview")
    fallback_prompt = "Sorry, I couldn’t find an answer that fits your context. Could you rephrase your question or provide more details?"

    # Initialize Conversation Chain
    conversation_chain = get_conversation_chain(retriever, llm_model, fallback_prompt)

    # Start Socket Server
    host = '127.0.0.1'
    port = 2004
    server_socket = socket.socket()
    server_socket.bind((host, port))
    server_socket.listen(1)
    logging.info(f"Server started at {host}:{port}. Waiting for connections.")

    sentiment_analyzer = SentimentIntensityAnalyzer()

    while True:
        conn, addr = server_socket.accept()
        logging.info(f"Connection established with {addr}")
        user_input = conn.recv(100000).decode("utf-8").strip()

        if user_input.lower() == "exit":
            logging.info("Server shutting down.")
            conn.close()
            break

        if "::" in user_input:
                student_state = user_input.split("::")[-1].strip().upper()
                selected_context = get_student_context(student_state) or global_context
                user_question = user_input.split("::")[0].strip()
                logging.info(f"Selected Context: {selected_context}")


        # Retrieve response based on context
        response = retrieve_response(user_question, faiss_index, id_to_row, selected_context, model)

        #if response == "No response found based on context.":
            # Fallback to LangChain
            #response = conversation_chain({'question': user_input})['answer']

        # Sentiment Analysis on User Input
        sentiment_score = sentiment_analyzer.polarity_scores(user_input)['compound']

        # Send Response
        response_message = f"{response}_{sentiment_score:.2f}"
        conn.send(response_message.encode())

    conn.close()

if __name__ == '__main__':
    main()

