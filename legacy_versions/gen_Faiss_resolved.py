import logging
import socket
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load and preprocess dataset
def load_data(data_path):
    """
    Load and preprocess the dataset. Combine relevant columns into a single text representation.
    """
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
    """
    Build a FAISS index for fast similarity search based on preprocessed data.
    """
    try:
        langchain_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Include row IDs as metadata during index creation
        metadata = [{"id": str(i)} for i in range(len(data))]
        faiss_index = FAISS.from_texts(
            texts=data['Combined'].tolist(),
            embedding=langchain_embeddings,
            metadatas=metadata
        )
        
        # Create a mapping of row IDs to data for retrieval
        id_to_row = data.to_dict(orient='index')
        return faiss_index, langchain_embeddings.client, id_to_row
    except Exception as e:
        logging.error(f"Failed to build FAISS index: {e}")
        raise


def retrieve_response(user_input, faiss_index, id_to_row, embedding_model, selected_context):
    """
    Retrieve the most contextually relevant response from the FAISS index.
    """
    try:
        print(selected_context)
        # Generate embedding for the user input
        user_embedding = embedding_model.encode([user_input])
        print(f"Shape of User Embedding: {user_embedding.shape}")
        # Perform a similarity search
        results = faiss_index.similarity_search_with_score_by_vector(user_embedding[0], k=115)
        print(f"Similarity Search Results: {results}")
        # Process results
        for document, score in results:
            row_id = int(document.metadata.get('id', -1))  # Safely retrieve the 'id'
            if row_id == -1:
                continue  # Skip if metadata is missing or invalid
            
            response_row = id_to_row[row_id]
            print("Response row")
            print(response_row)
            if (
                response_row['Understanding'].strip() == selected_context['Understanding'] and
                response_row['Emotion'].strip() == selected_context['Emotion'] and
                response_row['Reasoning'].strip() == selected_context['Reasoning']
            ):
                logging.info(f"Matched response found based on context. Score: {score}")
                return response_row['Response']
        logging.info("No matching response found based on context.")
        return "No response found based on context."  # No suitable match found
    except Exception as e:
        logging.error(f"Failed to retrieve response: {e}")
        return "No response found based on context."



# Retrieve context based on student state
def get_student_context(student_code):
    context_map = {
        "ED": {"Understanding": "High understanding", "Emotion": "engaged", "Reasoning": "deductive"},
        # Add other mappings...
    }
    return context_map.get(student_code.upper(), None)

# Start the server
def start_server(data_path, host='127.0.0.1', port=2004):
    try:
        data = load_data(data_path)
        faiss_index, embedding_model, id_to_row = build_faiss_index(data)
        global_context = {"Understanding": "High understanding", "Emotion": "engaged", "Reasoning": "deductive"}
        selected_context = global_context

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

            if user_question.lower() == "exit":
                logging.info("Exiting server.")
                conn.close()
                break

            if user_question:
                answer = retrieve_response(user_question, faiss_index, id_to_row, embedding_model, selected_context)

                # Sentiment analysis
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
    data_path = "./train_docs/Updated_Extracted_Data.csv"  # Replace with the actual path
    start_server(data_path)

