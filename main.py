# main.py - Intelligent Conversational Agent Backend

import os
import json
import fitz  # Using PyMuPDF for robust PDF reading
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# --- FLASK APP INITIALIZATION & SECURITY ---
app = Flask(__name__)
load_dotenv()

# CRITICAL SECURITY: Load API key from an environment variable.
# API_KEY = os.getenv("GOOGLE_API_KEY")
# if not API_KEY:
#     raise ValueError("SECURITY ALERT: Your GOOGLE_API_KEY is not set in the environment variables!")
genai.configure(api_key="AIzaSyBpng2PGXkS_3i263HO8Rqu0Qk6-IAfWLg")

# --- CONFIGURATION ---
UPLOAD_FOLDER = "policy_documents"
VECTOR_STORE_DIR = "faiss_vector_stores"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# --- GLOBAL EMBEDDING MODEL (Initialized once) ---
print("Initializing embedding model...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
print("Embedding model loaded.")


# --- DOCUMENT PROCESSING (with robust PDF handling) ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(file_path, filename):
    file_text = ""
    try:
        if filename.endswith(".pdf"):
            with fitz.open(file_path) as doc:
                for page in doc:
                    file_text += page.get_text() + "\n"
        # Add other file types as needed
        elif filename.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                file_text = f.read()
        elif filename.endswith(".docx"):
            doc = docx.Document(file_path)
            file_text = "\n".join([para.text for para in doc.paragraphs])
            
        print(f"Successfully extracted text from: {filename}")
        return file_text
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return None

# --- VECTOR STORE & RETRIEVAL LOGIC ---
def create_and_save_vector_store(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    store_path = os.path.join(VECTOR_STORE_DIR, f"{filename}.faiss")
    if os.path.exists(store_path):
        return True

    document_text = extract_text_from_file(file_path, filename)
    if not document_text or not document_text.strip():
        print(f"ERROR: No text extracted from {filename}.")
        return False

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.split_text(document_text)
    if not chunks:
        return False

    store = FAISS.from_texts(texts=chunks, embedding=embeddings)
    store.save_local(store_path)
    return True

def get_combined_retriever(selected_policy):
    main_store_path = os.path.join(VECTOR_STORE_DIR, f"{selected_policy}.faiss")
    if not os.path.exists(main_store_path) and not create_and_save_vector_store(selected_policy):
        return None
    try:
        main_store = FAISS.load_local(main_store_path, embeddings, allow_dangerous_deserialization=True)
    except Exception:
        return None

    # Merge other available stores for a comprehensive search
    other_stores = [f for f in os.listdir(VECTOR_STORE_DIR) if f.endswith('.faiss') and f != f"{selected_policy}.faiss"]
    for store_file in other_stores:
        try:
            other_store_path = os.path.join(VECTOR_STORE_DIR, store_file)
            main_store.merge_from(FAISS.load_local(other_store_path, embeddings, allow_dangerous_deserialization=True))
        except Exception:
            continue
    return main_store.as_retriever(search_kwargs={"k": 5})

# --- INTELLIGENT CONVERSATIONAL AI ---
def get_intelligent_chat_response(chat_history: list, relevant_clauses: str, user_query: str):
    """Generates a conversational response that includes a follow-up question."""
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    history_string = "\n".join([f"{entry['role']}: {entry['content']}" for entry in chat_history])

    prompt = f"""
    You are a highly intelligent and empathetic insurance policy assistant. Your goal is to provide clear answers and guide the user to the information they need.

    **Conversation Context:**
    - **History:**
    {history_string}
    - **Relevant Policy Clauses:**
    ---
    {relevant_clauses}
    ---

    **User's Latest Message:** "{user_query}"

    **Your Task:**
    Respond in a specific JSON format. Your response must be a single JSON object with two keys: "answer" and "follow_up_question".

    1.  **"answer"**: Formulate a direct, helpful answer to the user's latest message based *only* on the provided clauses and history.
    2.  **"follow_up_question"**: After providing the answer, ask a thoughtful follow-up question to guide the conversation.
        - If the user's query was vague, ask for more specific details (e.g., "For what specific procedure do you need coverage information?").
        - If you provided a complete answer, ask if the user is satisfied or needs more help (e.g., "Does this clarify your question, or is there another part of the policy you'd like to explore?").
        - If you couldn't find an answer, state that clearly in the "answer" field and ask the user if they'd like you to search for something else in the "follow_up_question".

    **Example JSON Response:**
    {{
      "answer": "Yes, the policy covers hospitalization expenses up to ₹5,00,000 per year.",
      "follow_up_question": "Were you interested in the coverage for room rent or for specific surgical procedures?"
    }}
    """
    
    try:
        # Requesting JSON output directly from the model
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        return json.loads(response.text)
    except Exception as e:
        print(f"Error processing chat with LLM: {e}")
        return {
            "answer": "I'm sorry, I'm having a little trouble right now. Could you please rephrase your question?",
            "follow_up_question": "You can also try selecting a different document."
        }

# --- WEB ROUTES ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload_policy', methods=['POST'])
def upload_policy():
    if 'policyFile' not in request.files: return jsonify({'error': 'No file part.'}), 400
    file = request.files['policyFile']
    if file.filename == '' or not allowed_file(file.filename): return jsonify({'error': 'Invalid file.'}), 400
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        file.save(save_path)
        if create_and_save_vector_store(filename):
            return jsonify({'message': f'Successfully indexed {filename}'}), 200
        else:
            return jsonify({'error': f'"{filename}" might be empty or unreadable.'}), 500
    except Exception as e: return jsonify({'error': f'Server error: {e}'}), 500

@app.route('/list_policies')
def list_policies():
    return jsonify({'policies': [f for f in os.listdir(UPLOAD_FOLDER) if allowed_file(f)]})

@app.route('/chat', methods=['POST'])
def chat_route():
    data = request.get_json()
    user_query, policy, history = data.get('query'), data.get('policy'), data.get('history', [])
    if not user_query or not policy: return jsonify({'error': 'Query and policy are required.'}), 400

    retriever = get_combined_retriever(policy)
    if not retriever: return jsonify({"error": "Selected document could not be loaded."}), 500

    clauses = "\n\n---\n\n".join([doc.page_content for doc in retriever.invoke(user_query)])
    bot_response = get_intelligent_chat_response(history, clauses, user_query)
    return jsonify(bot_response)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("--- Starting Intelligent Policy Assistant Server ---")
    print("Indexing existing documents...")
    for filename in os.listdir(UPLOAD_FOLDER):
        if allowed_file(filename): create_and_save_vector_store(filename)
    print(f"--- Server ready at http://127.0.0.1:5000 ---")
    app.run(host='0.0.0.0', port=5000, debug=False)