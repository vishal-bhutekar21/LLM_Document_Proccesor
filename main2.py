import os
import json
import PyPDF2
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai

# --- CONFIGURATION ---

API_KEY = "AIzaSyBpng2PGXkS_3i263HO8Rqu0Qk6-IAfWLg"
genai.configure(api_key=API_KEY)


DOCUMENTS_DIR = "policy_documents" 
VECTOR_STORE_PATH = "vector_store"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- NEW FUNCTION: ADVANCED DOCUMENT LOADER ---

def load_documents_from_directory(directory_path):
    """
    Loads and extracts text from all supported files (txt, pdf, docx)
    in a given directory.
    """
    all_texts = []
    print(f"--- Loading documents from '{directory_path}' ---")
    
    if not os.path.exists(directory_path):
        print(f"Error: Directory not found at '{directory_path}'")
        # Create the directory and a sample file for the user
        os.makedirs(directory_path)
        with open(os.path.join(directory_path, "sample_policy.txt"), "w") as f:
            f.write("This is a sample policy. Accidents are covered from day 1.")
        print(f"Created a sample directory and file. Please add your documents there and restart.")
        return []

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        file_text = ""
        try:
            if filename.endswith(".pdf"):
                with open(file_path, 'rb') as pdf_file:
                    reader = PyPDF2.PdfReader(pdf_file)
                    for page in reader.pages:
                        file_text += page.extract_text() + "\n"
                print(f"Successfully loaded PDF: {filename}")

            elif filename.endswith(".docx"):
                doc = docx.Document(file_path)
                for para in doc.paragraphs:
                    file_text += para.text + "\n"
                print(f"Successfully loaded DOCX: {filename}")

            elif filename.endswith(".txt"):
                with open(file_path, 'r', encoding='utf-8') as txt_file:
                    file_text = txt_file.read()
                print(f"Successfully loaded TXT: {filename}")
            
            if file_text:
                all_texts.append(file_text)

        except Exception as e:
            print(f"Could not read file {filename}. Error: {e}")
            
    return all_texts

# --- UPDATED FUNCTION: CREATE VECTOR STORE ---

def create_vector_store():
    """
    Loads documents from a directory, splits them into chunks, 
    creates embeddings, and saves them to a FAISS vector store.
    """
    print("--- Starting document indexing ---")
    
    # 1. Load all documents from the directory
    documents = load_documents_from_directory(DOCUMENTS_DIR)
    if not documents:
        print("No documents were loaded. Exiting indexing process.")
        return None

    # 2. Split the documents into chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.create_documents(documents) # Use create_documents for list of texts
    print(f"Created {len(chunks)} chunks from all documents.")

    # 3. Initialize the embedding model
    print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # 4. Create the FAISS vector store from the chunks
    print("Creating vector store...")
    # Use from_documents since we now have a list of LangChain Document objects
    vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)

    # 5. Save the vector store locally
    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"--- Document indexing complete. Vector store saved to: {VECTOR_STORE_PATH} ---")
    return vector_store

# --- (The rest of the functions remain the same) ---

def parse_query_with_llm(user_query: str):
    """
    Uses the Gemini LLM to parse the user's natural language query
    into a structured JSON object.
    """
    print("\n--- Parsing user query with LLM ---")
    
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    prompt = f"""
    You are an expert at parsing insurance claim details. Your task is to extract key entities from the user's query.

    Please extract the following information:
    - 'age': The age of the person (as an integer).
    - 'procedure': The medical procedure or surgery mentioned (as a string).
    - 'location': The city or location mentioned (as a string).
    - 'policy_duration_months': The age of the policy in months (as an integer).

    Respond ONLY with a JSON object. If a value for a key is not found in the query, set its value to null.

    User Query: "{user_query}"
    """
    
    try:
        response = model.generate_content(prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
        print(f"LLM Parsed Details: {cleaned_response}")
        return json.loads(cleaned_response)
    except Exception as e:
        print(f"Error parsing query with LLM: {e}")
        return None

def retrieve_relevant_clauses(query: str, vector_store):
    """
    Searches the vector store for the most relevant document chunks (clauses)
    based on the user's query.
    """
    print("\n--- Retrieving relevant clauses from vector store ---")
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    relevant_docs = retriever.invoke(query)
    
    clauses_text = "\n\n".join([doc.page_content for doc in relevant_docs])
    print("Found relevant clauses.")
    return clauses_text

def get_decision_from_llm(parsed_details: dict, relevant_clauses: str):
    """
    Uses the Gemini LLM to make a final decision based on the parsed query
    and the retrieved policy clauses.
    """
    print("\n--- Getting final decision from LLM (Reasoning Engine) ---")
    
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    prompt = f"""
    You are a senior insurance claim adjudicator. Your task is to make a decision on a claim based on the claimant's details and the relevant policy clauses provided.

    **Claimant Details (in JSON format):**
    {json.dumps(parsed_details, indent=2)}

    **Relevant Policy Clauses:**
    ---
    {relevant_clauses}
    ---

    **Your Instructions:**
    1.  Carefully analyze the "Claimant Details" against the "Relevant Policy Clauses".
    2.  Determine if the claim should be "Approved" or "Rejected".
    3.  If the claim is rejected, clearly state the reason and cite the specific clause number (e.g., "Rejected based on Clause 2.2").
    4.  If the claim is approved, state the covered amount if it can be determined from the clauses.
    
    **Provide your final answer ONLY as a single, clean JSON object with three keys:**
    - "decision": Your final decision, either "Approved" or "Rejected".
    - "amount": The approved amount as an integer (use 0 if rejected or if the amount cannot be determined).
    - "justification": A clear and concise explanation for your decision, referencing the specific clause number(s).
    """

    try:
        response = model.generate_content(prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
        print("--- Final Decision Received ---")
        return json.loads(cleaned_response)
    except Exception as e:
        print(f"Error getting decision from LLM: {e}")
        return None


# --- MAIN EXECUTION ---

def main():
    # Check if vector store exists. If not, create it.
    # This will now process all files in the "policy_documents" directory.
    if not os.path.exists(VECTOR_STORE_PATH):
        if create_vector_store() is None:
            # If no documents were found and the sample was created, exit.
            print("\nPlease add your documents to the 'policy_documents' folder and run the script again.")
            return

    # Load the existing vector store
    print("\nLoading existing vector store...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    print("Vector store loaded successfully.")

    # --- Start the interactive query loop ---
    print("\n\n--- LLM Document Processing System Ready ---")
    print("Enter your query below, or type 'exit' to quit.")
    
    while True:
        user_query = input("\nYour Query: ")
        if user_query.lower() == 'exit':
            break

        # 1. Parse the query
        parsed_details = parse_query_with_llm(user_query)
        if not parsed_details:
            print("Could not process the query. Please try again.")
            continue

        # 2. Retrieve relevant clauses
        relevant_clauses = retrieve_relevant_clauses(user_query, vector_store)

        
        final_decision = get_decision_from_llm(parsed_details, relevant_clauses)
        
        if final_decision:
            print("\n================ FINAL RESULT ================")
            print(json.dumps(final_decision, indent=4))
            print("============================================")
        else:
            print("Could not get a final decision. Please check the logs.")


if __name__ == "__main__":
    main()
