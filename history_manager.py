import os
import json
from datetime import datetime

# Configuration for the history log file
HISTORY_LOG_FILE = "claim_history.json"

def read_history():
    """
    Safely reads the claim history from the JSON log file.
    Returns an empty list if the file doesn't exist or is invalid.
    """
    if not os.path.exists(HISTORY_LOG_FILE):
        return []
    try:
        with open(HISTORY_LOG_FILE, 'r', encoding='utf-8') as f:
            # Handle empty file case
            content = f.read()
            if not content:
                return []
            return json.loads(content)
    except (json.JSONDecodeError, IOError):
        # If the file is corrupted or unreadable, return an empty list to prevent crashes
        return []

def write_history(data):
    """
    Writes the provided data to the JSON log file.
    """
    try:
        with open(HISTORY_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
    except IOError as e:
        print(f"Error writing to history file: {e}")

def add_claim_to_history(user_query, policy, adjudication_result):
    """
    A complete function to add a new, adjudicated claim to the history log.
    This function reads the existing history, appends the new record, and writes it back.
    """
    if adjudication_result.get("decision") == "Error":
        # Do not log entries that resulted in an error
        return

    claim_history = read_history()

    # Create a new, timestamped entry for the history log
    history_entry = {
        "id": len(claim_history) + 1,
        "query": user_query,
        "decision": adjudication_result.get("decision"),
        "amount": adjudication_result.get("amount"),
        "justification": adjudication_result.get("justification"),
        "policy_document": policy,
        "timestamp": datetime.now().isoformat()
    }

    # Add the new claim to the beginning of the list (most recent first)
    claim_history.insert(0, history_entry)
    
    # Save the updated history back to the file
    write_history(claim_history)