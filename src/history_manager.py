
import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

HISTORY_FILE = "history.json"

def _get_history_path():
    """
    Constructs the absolute path to the history file, ensuring it's in the project root.
    The project root is assumed to be the parent directory of the 'src' directory.
    """
    # __file__ is the path to this file (src/history_manager.py)
    # os.path.dirname(__file__) is the 'src' directory
    # os.path.dirname(os.path.dirname(__file__)) is the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, HISTORY_FILE)

def load_history() -> List[Dict[str, Any]]:
    """Loads the history from the JSON file."""
    history_path = _get_history_path()
    if not os.path.exists(history_path):
        return []
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            # Handle empty file case
            content = f.read()
            if not content:
                return []
            return json.loads(content)
    except (json.JSONDecodeError, IOError):
        return []

def save_history(history: List[Dict[str, Any]]):
    """Saves the entire history list to the JSON file."""
    history_path = _get_history_path()
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def add_history_entry(problem: str, mode: str, solutions: int, source: str) -> Dict[str, Any]:
    """Creates and adds a new history entry."""
    history = load_history()
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "problem": problem,
        "mode": mode,
        "solutions": solutions,
        "source": source, # 'cli' or 'web'
        "status": "running",
        "final_review": None,
        "usage": None,
        "graph_data": None
    }
    history.insert(0, entry) # Add to the top
    save_history(history)
    return entry

def update_history_entry(entry_id: str, updates: Dict[str, Any]):
    """Updates a specific history entry by its ID."""
    history = load_history()
    entry_found = False
    for entry in history:
        if entry.get("id") == entry_id:
            entry.update(updates)
            entry_found = True
            break
    if entry_found:
        save_history(history)
    else:
        # This can happen in multi-threaded/process environments if the file was modified
        # between load and save. For this simple app, we'll just log it.
        print(f"Warning: Could not find history entry with ID {entry_id} to update.")
