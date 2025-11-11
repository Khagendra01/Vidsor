"""
Chat history management for Vidsor.
"""

import os
import json
from typing import List, Dict, Optional


class ChatManager:
    """Handles chat history loading and saving."""
    
    @staticmethod
    def load_chat_history(project_path: Optional[str]) -> List[Dict[str, str]]:
        """
        Load chat history from project folder.
        
        Args:
            project_path: Path to project folder
            
        Returns:
            List of chat messages (dicts with "role" and "content")
        """
        if not project_path:
            return []
        
        chat_history_path = os.path.join(project_path, "chat_history.json")
        if not os.path.exists(chat_history_path):
            return []
        
        try:
            # Check if file is empty or whitespace only
            with open(chat_history_path, 'r') as f:
                content = f.read().strip()
                if not content:
                    print("[VIDSOR] chat_history.json is empty, starting with empty chat history")
                    return []
            
            # Parse JSON
            with open(chat_history_path, 'r') as f:
                chat_data = json.load(f)
            
            # Validate structure - should be a list
            if not isinstance(chat_data, list):
                print("[VIDSOR] chat_history.json has invalid structure (expected list), starting with empty chat history")
                return []
            
            # Validate each entry is a dict with required keys
            valid_history = []
            for msg in chat_data:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    valid_history.append(msg)
                else:
                    print(f"[VIDSOR] Skipping invalid chat history entry: {msg}")
            
            if len(valid_history) != len(chat_data):
                print(f"[VIDSOR] Loaded {len(valid_history)} valid entries from chat_history.json (skipped {len(chat_data) - len(valid_history)} invalid)")
            elif len(valid_history) > 0:
                print(f"[VIDSOR] Loaded {len(valid_history)} chat history entries")
            
            return valid_history
            
        except json.JSONDecodeError as e:
            print(f"[VIDSOR] chat_history.json contains invalid JSON, starting with empty chat history")
            return []
        except Exception as e:
            print(f"[VIDSOR] Failed to load chat history: {e}")
            return []
    
    @staticmethod
    def save_chat_history(project_path: Optional[str], chat_history: List[Dict[str, str]]) -> None:
        """
        Save chat history to project folder.
        
        Args:
            project_path: Path to project folder
            chat_history: List of chat messages to save
        """
        if not project_path:
            return
        
        chat_history_path = os.path.join(project_path, "chat_history.json")
        try:
            with open(chat_history_path, 'w') as f:
                json.dump(chat_history, f, indent=2)
        except Exception as e:
            print(f"[VIDSOR] Failed to save chat history: {e}")

