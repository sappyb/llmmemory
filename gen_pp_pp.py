from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import socket

# Load and preprocess dataset
data_path = "./train_docs/Updated_Extracted_Data.csv"  # Replace with the actual path
data = pd.read_csv(data_path)

# Combine relevant fields into one text column
data['Combined'] = (
    "Question: " + data['Question'] +
    " Understanding: " + data['Understanding'] +
    " Emotion: " + data['Emotion'] +
    " Reasoning: " + data['Reasoning'] +
    " Response: " + data['Response']
)

# Generate embeddings and build FAISS index
def build_faiss_index(data):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(data['Combined'].tolist())
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)
    id_to_row = {i: row for i, row in data.iterrows()}
    return faiss_index, model, id_to_row

# Context Mapping based on Student State
def get_student_context(student_code):
    context_map = {
        "ED": {"Understanding": "High understanding", "Emotion": "engaged", "Reasoning": "deductive"},
        "BD": {"Understanding": "High understanding", "Emotion": "bored", "Reasoning": "deductive"},
        "AD": {"Understanding": "High understanding", "Emotion": "anxious", "Reasoning": "deductive"},
        "DD": {"Understanding": "High understanding", "Emotion": "distressed", "Reasoning": "deductive"},
        "FD": {"Understanding": "High understanding", "Emotion": "fatigued", "Reasoning": "deductive"},
        "FED": {"Understanding": "High understanding", "Emotion": "fedup", "Reasoning": "deductive"},
        "EA": {"Understanding": "High understanding", "Emotion": "engaged", "Reasoning": "analogical"},
        "BA": {"Understanding": "High understanding", "Emotion": "bored", "Reasoning": "analogical"},
        "AA": {"Understanding": "High understanding", "Emotion": "anxious", "Reasoning": "analogical"},
        "DA": {"Understanding": "High understanding", "Emotion": "distressed", "Reasoning": "analogical"},
        "FA": {"Understanding": "High understanding", "Emotion": "fatigued", "Reasoning": "analogical"},
        "FEA": {"Understanding": "High understanding", "Emotion": "fedup", "Reasoning": "analogical"},
    }
    return context_map.get(student_code.upper(), None)

# Retrieve response
def retrieve_response(user_input, selected_context, faiss_index, model, id_to_row):
    user_embedding = model.encode([user_input])
    k = 229  # Adjust as needed
    _, indices = faiss_index.search(user_embedding, k=k)

    for row_id in indices[0]:
        response_row = id_to_row[row_id]
        if (
            response_row['Understanding'].strip() == selected_context['Understanding'] and
            response_row['Emotion'].strip() == selected_context['Emotion'] and
            response_row['Reasoning'].strip() == selected_context['Reasoning']
        ):
            return response_row['Response']

    return "Sorry, I couldn't find a suitable response."

# Main function
def main():
    # Build FAISS index
    faiss_index, model, id_to_row = build_faiss_index(data)

    # Set default context
    global_student_type = {"Understanding": "High understanding", "Emotion": "engaged", "Reasoning": "deductive"}
    selected_context = global_student_type

    # Start socket server
    host = '127.0.0.1'
    port = 2004
    s = socket.socket()
    s.bind((host, port))
    s.listen(1)
    print("Server started... Waiting for connections.")

    while True:
        conn, addr = s.accept()
        user_question = conn.recv(100000).decode("utf-8").strip()

        if "::" in user_question:
            student_state = user_question.split("::")[-1].strip().upper()
            selected_context = get_student_context(student_state) or global_student_type
            user_question = user_question.split("::")[0].strip()

        if user_question.lower() == "exit":
            print("Goodbye!")
            conn.close()
            break

        if user_question:
            answer = retrieve_response(user_question, selected_context, faiss_index, model, id_to_row)
            
            # Sentiment analysis
            sid_obj = SentimentIntensityAnalyzer()
            sentiment_score = sid_obj.polarity_scores(user_question)['compound']

            print(f"Question: {user_question}")
            print(f"Response: {answer}")
            print(f"Sentiment Score: {sentiment_score:.2f}")

            response_final = f"{answer}_{sentiment_score:.2f}"
            conn.send(response_final.encode())

    conn.close()

if __name__ == '__main__':
    main()

