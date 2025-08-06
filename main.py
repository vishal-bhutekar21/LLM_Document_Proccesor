# main.py - Intelligent Claims Adjudicator Backend

import os
import json
import fitz  # PyMuPDF
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

# CRITICAL SECURITY: Load API key from environment variables.
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("SECURITY ALERT: GOOGLE_API_KEY is not set in environment variables!")
genai.configure(api_key=API_KEY)

# --- CONFIGURATION ---
UPLOAD_FOLDER = "policy_documents"
VECTOR_STORE_DIR = "faiss_vector_stores"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# --- GLOBAL EMBEDDING MODEL ---
print("Initializing embedding model...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
print("Embedding model loaded.")

# --- DOCUMENT PROCESSING LOGIC ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(file_path, filename):
    # This function remains the same, effectively handling PDF, DOCX, and TXT
    # ... (code from previous response) ...
    file_text = ""
    try:
        if filename.endswith(".pdf"):
            with fitz.open(file_path) as doc:
                file_text = "\n".join([page.get_text() for page in doc])
        elif filename.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                file_text = f.read()
        elif filename.endswith(".docx"):
            doc = docx.Document(file_path)
            file_text = "\n".join([para.text for para in doc.paragraphs])
        return file_text
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return None

# --- VECTOR STORE & RETRIEVAL LOGIC ---
def create_and_save_vector_store(filename):
    # This function remains the same
    # ... (code from previous response) ...
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    store_path = os.path.join(VECTOR_STORE_DIR, f"{filename}.faiss")
    if os.path.exists(store_path): return True
    document_text = extract_text_from_file(file_path, filename)
    if not document_text or not document_text.strip(): return False
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_text(document_text)
    if not chunks: return False
    store = FAISS.from_texts(texts=chunks, embedding=embeddings)
    store.save_local(store_path)
    return True

def get_retriever(selected_policy):
    # This function remains the same
    # ... (code from previous response) ...
    store_path = os.path.join(VECTOR_STORE_DIR, f"{selected_policy}.faiss")
    if not os.path.exists(store_path):
        if not create_and_save_vector_store(selected_policy): return None
    try:
        vector_store = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
        return vector_store.as_retriever(search_kwargs={"k": 7})
    except Exception as e:
        print(f"Error loading vector store {store_path}: {e}")
        return None

# --- INTELLIGENT DECISION-MAKING AI ---
def get_adjudication_from_llm(chat_history: list, relevant_clauses: str, user_query: str):
    """
    Instructs the LLM to parse a query, evaluate it against clauses, and return a structured JSON decision.
    """
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    
    # *** KEY IMPROVEMENT BASED ON YOUR PROBLEM STATEMENT ***
    # The prompt now explicitly asks the model to parse the query first.
    prompt = f"""
    You are a meticulous Insurance Claims Adjudicator AI. Your task is to analyze a user's query against policy clauses and render a precise, evidence-based judgment.

    **Analysis Context:**
    - **Conversation History:** {chat_history}
    - **User's Current Query:** "{user_query}"
    - **Relevant Policy Clauses (Source of Truth):**
    ---
    {relevant_clauses}
    ---

    **Your Mandate (Follow these steps precisely):**
    1.  **Parse Query:** First, deconstruct the user's query to identify all key details. Extract entities like age, gender, medical procedure, location, policy duration, etc.
    2.  **Evaluate Against Clauses:** Systematically compare the parsed details from the query against the rules in the "Relevant Policy Clauses".
    3.  **Decide and Justify:** Based on the evaluation, make a final decision. Your justification must clearly explain *how* the policy rules apply to the user's specific situation.
    4.  **Cite Sources:** You MUST quote the exact clause(s) from the provided text that are the basis for your decision. This is non-negotiable for audit purposes.

    **Output Format:**
    Your final output MUST be a single, valid JSON object. Do not include any text outside of this JSON structure.
    {{
      "decision": "string (e.g., 'Approved', 'Rejected', 'More Information Needed')",
      "amount": "string (e.g., '₹50,000') or null",
      "justification": "string (A clear explanation for the decision, referencing the user's details and policy rules).",
      "cited_clauses": [
        "string (Direct quote of the first supporting clause)",
        "string (Direct quote of the second supporting clause, if any)"
      ]
    }}

    **Example:**
    - User Query: "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
    - Relevant Clauses: "1. A mandatory waiting period of 24 months is applicable for joint replacement surgeries. 2. All surgeries are covered up to ₹2,00,000."
    - Expected JSON Output:
    {{
      "decision": "Rejected",
      "amount": null,
      "justification": "The claim for knee surgery is rejected because the policy has a mandatory 24-month waiting period for this procedure. The user's policy is only 3 months old, which is within the waiting period.",
      "cited_clauses": [
        "A mandatory waiting period of 24 months is applicable for joint replacement surgeries."
      ]
    }}

    Now, process the user's query and provide your adjudication in the specified JSON format.
    """
    
    try:
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Error processing adjudication with LLM: {e}")
        return {
            "decision": "Error", "amount": None,
            "justification": "An internal error occurred. Please try rephrasing your question or check the document.",
            "cited_clauses": []
        }

# --- WEB ROUTES & MAIN EXECUTION ---
# All routes (/, /upload_policy, /list_policies, /chat) and the main execution block
# remain the same as the previous version. They are already set up correctly.
# ... (code from previous response) ...
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload_policy', methods=['POST'])
def upload_policy():
    if 'policyFile' not in request.files: return jsonify({'error': 'No file part.'}), 400
    file = request.files['policyFile']
    if file.filename == '' or not allowed_file(file.filename): return jsonify({'error': 'Invalid file type.'}), 400
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        file.save(save_path)
        if create_and_save_vector_store(filename):
            return jsonify({'message': f'Successfully indexed {filename}', 'filename': filename}), 200
        else:
            os.remove(save_path)
            return jsonify({'error': f'The uploaded file "{filename}" appears to be empty or could not be read.'}), 500
    except Exception as e:
        return jsonify({'error': f'Server error during file upload: {e}'}), 500

@app.route('/list_policies')
def list_policies():
    try:
        policies = [f for f in os.listdir(UPLOAD_FOLDER) if allowed_file(f)]
        return jsonify({'policies': policies})
    except Exception as e:
        return jsonify({'error': f'Could not list policies: {e}'}), 500

@app.route('/chat', methods=['POST'])
def chat_route():
    data = request.get_json()
    user_query, policy, history = data.get('query'), data.get('policy'), data.get('history', [])
    if not all([user_query, policy]):
        return jsonify({'error': 'A query and a selected policy are required.'}), 400

    retriever = get_retriever(policy)
    if not retriever:
        return jsonify({"error": "The selected policy document could not be loaded or processed."}), 500

    retrieved_docs = retriever.invoke(user_query)
    clauses = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
    if not clauses.strip():
        clauses = "No relevant clauses were found in the document for this query."

    bot_response_json = get_adjudication_from_llm(history, clauses, user_query)
    return jsonify(bot_response_json)

if __name__ == "__main__":
    print("--- Starting Intelligent Policy Adjudicator Server ---")
    print("Indexing any existing documents...")
    for filename in os.listdir(UPLOAD_FOLDER):
        if allowed_file(filename):
            create_and_save_vector_store(filename)
    print(f"--- Server ready. Access at http://127.0.0.1:5000 ---")
    app.run(host='0.0.0.0', port=5000, debug=True)