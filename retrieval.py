import numpy as np
import time
import fitz  # PyMuPDF
import openai
import faiss
import re
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Initialize FAISS Index for Vectors
dimension = 1536  # Dimension of the embeddings
index = faiss.IndexFlatL2(dimension)
metadata_pd = []

# Initialize FAISS Index for Vectors
dimension = 1536  # Dimension of the embeddings
index_recall = faiss.IndexFlatL2(dimension)
metadata = []

def store_pdf(user_id, file_name, file_path):
    with fitz.open(file_path) as doc:
        content = ""
        for page in doc:
            content += page.get_text()
    timestamp = time.time()

    vector = vectorize_text(content)
    metadata_pd.append({
        "user_id": user_id,
        "content": content,
        "timestamp": timestamp,
    })
    index.add(np.array([vector]))

def search_pdfs(user_id, input_vec):
    k = 2  # Number of nearest neighbors to search
    distances, indices = index.search(np.array([input_vec]), k)
    results = []
    for idx in indices[0]:
        if idx != -1 and metadata_pd[idx]["user_id"] == user_id:
            results.append(metadata_pd[idx]["content"])
    return results

# Define Memory Recall and Consolidation Functions
def vectorize_text(text):
    response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
    return np.array(response['data'][0]['embedding'])

# Check if a text is a question
def is_question(text):
    question_words = {"what", "why", "how", "is", "are", "was", "were", "do", "does", "did", "can", "could", "should", "would", "which", "who", "whom", "whose", "where", "when"}
    words = text.lower().split()
    if words and words[0] in question_words:
        return True
    return re.search(r'\?\s*$', text.strip()) is not None

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def calculate_relevance(memory_vec, input_vec):
    return cosine_similarity(memory_vec, input_vec)

def exponential_decay(relevance, elapsed_time, decay_rate):
    return relevance * np.exp(-decay_rate * elapsed_time)

def recall_probability(relevance, elapsed_time, decay_rate):
    return 1 - np.exp(-exponential_decay(relevance, elapsed_time, decay_rate))

def recall_memory(user_id, input_vec, threshold=0.10, decay_rate=0.001):
    current_time = time.time()
    if len(metadata) == 0:
        return None
    distances, indices = index_recall.search(np.array([input_vec]), len(metadata))
    print("Indices:", indices)
    print("Distances:", distances)
    for idx in indices[0]:
        print(idx)
        if idx == -1:
            continue
        memory_vec = index_recall.reconstruct(int(idx))
        relevance = calculate_relevance(memory_vec, input_vec)
        elapsed_time = current_time - metadata[idx]["timestamp"]
        prob = recall_probability(relevance, elapsed_time, decay_rate)
        print("Recall Memory:", metadata[idx]["content"])
        print("Probability:", prob)
        print("Relevance:", relevance)
        print("Elapsed Time:", elapsed_time)
        if prob > threshold:
            print("Recall Memory:", metadata[idx]["content"])
            return metadata[idx]["content"]
    return None

# Memory Storage
def store_memory(user_id, content):
    if is_question(content):
        print("Not storing question:", content)
        return
    print("Storing memory for user:", user_id)
    vector = vectorize_text(content)
    print("Shape of vector:", vector.shape)
    timestamp = time.time()
    metadata.append({
        "user_id": user_id,
        "content": content,
        "timestamp": timestamp,
    })
    index_recall.add(np.array([vector]))

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

# Chatbot Interaction
def generate_response(user_id, input_text):
    input_vec = vectorize_text(input_text)
    recalled_memory = recall_memory(user_id, input_vec)
    print("Recalled Memory:", recalled_memory)
    pdf_contents = search_pdfs(user_id, input_vec)
    print (pdf_contents)
    # Summarize PDF content if it's too long
    summarized_pdf_contents = [summarize_text(content) for content in pdf_contents]
    combined_pdf_content = "\n".join(summarized_pdf_contents)  # Full content for response
    if recalled_memory != None:
      combined_input = input_text + '\n Answer words : ' + recalled_memory
    else:
      # Combine inputs
      combined_input = input_text + '\n Answer words : ' + combined_pdf_content

    # Calculate the available tokens for input_text and truncate accordingly
    max_total_tokens = 4097
    reserved_completion_tokens = 256
    max_input_tokens = max_total_tokens - reserved_completion_tokens

    combined_input = truncate_text(combined_input, max_input_tokens)
    print("Combined Input:", combined_input)

    memory = ConversationBufferMemory()
    llm = OpenAI(api_key=openai.api_key)
    chain = ConversationChain(llm=llm, memory=memory)

    response = chain.run(input=combined_input)
    store_memory(user_id, input_text)
    return response