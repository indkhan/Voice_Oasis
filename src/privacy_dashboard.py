"""
Privacy Dashboard - A standalone application for Helbling VoiceBot privacy controls

This module provides a completely separate web application that allows users to view and delete
their stored voice data, without integrating with or modifying the main relay.py application.
"""

import os
import json
import pickle
import flask
import sys
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS

# Create standalone Flask app
app = Flask(__name__)
CORS(app)


# Find the voice_database directory
def find_voice_database():
    """Find the voice_database directory by searching up the directory tree"""
    # Start with the current directory
    current_dir = os.getcwd()

    # First, check if we're in the src directory
    if os.path.basename(current_dir) == "src":
        # Check for voice_database one level up
        parent_dir = os.path.dirname(current_dir)
        potential_path = os.path.join(parent_dir, "voice_database")
        if os.path.exists(potential_path):
            return potential_path

    # Next, check if we're already in the project root with voice_database
    potential_path = os.path.join(current_dir, "voice_database")
    if os.path.exists(potential_path):
        return potential_path

    # Try one more level up (if we're in a subdirectory)
    parent_dir = os.path.dirname(current_dir)
    potential_path = os.path.join(parent_dir, "voice_database")
    if os.path.exists(potential_path):
        return potential_path

    # Try using the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Check script directory
    potential_path = os.path.join(script_dir, "voice_database")
    if os.path.exists(potential_path):
        return potential_path

    # Check one level up from script directory
    parent_dir = os.path.dirname(script_dir)
    potential_path = os.path.join(parent_dir, "voice_database")
    if os.path.exists(potential_path):
        return potential_path

    # Fallback to default
    return "voice_database"


# Find the database directory
DB_DIR = find_voice_database()
EMBEDDINGS_FILE = os.path.join(DB_DIR, "embeddings.pkl")
MEMORIES_FILE = os.path.join(DB_DIR, "memories.json")
CHAT_SESSIONS_FILE = os.path.join(DB_DIR, "chat_sessions.json")

print(f"Using database directory: {DB_DIR}")
print(f"Memories file path: {MEMORIES_FILE}")
print(f"Memories file exists: {os.path.exists(MEMORIES_FILE)}")


def load_databases():
    """Load voice embeddings and memories from files"""
    voice_embeddings = {}
    voice_memories = {}
    chat_sessions = {}

    # Load embeddings
    if os.path.exists(EMBEDDINGS_FILE):
        try:
            with open(EMBEDDINGS_FILE, "rb") as f:
                voice_embeddings = pickle.load(f)
            print(f"Loaded {len(voice_embeddings)} voice embeddings")
        except Exception as e:
            print(f"Error loading embeddings: {e}")

    # Load memories
    if os.path.exists(MEMORIES_FILE):
        try:
            with open(MEMORIES_FILE, "r") as f:
                voice_memories = json.load(f)
            print(
                f"Loaded memories for {len(voice_memories)} speakers: {list(voice_memories.keys())}"
            )
        except Exception as e:
            print(f"Error loading memories: {e}")
    else:
        print(f"Memories file not found at: {MEMORIES_FILE}")

    # Load chat sessions
    if os.path.exists(CHAT_SESSIONS_FILE):
        try:
            with open(CHAT_SESSIONS_FILE, "r") as f:
                chat_sessions = json.load(f)
            print(f"Loaded {len(chat_sessions)} chat sessions")
        except Exception as e:
            print(f"Error loading chat sessions: {e}")

    return voice_embeddings, voice_memories, chat_sessions


def save_databases(voice_embeddings, voice_memories, chat_sessions):
    """Save databases back to files"""
    # Save embeddings
    try:
        with open(EMBEDDINGS_FILE, "wb") as f:
            pickle.dump(voice_embeddings, f)
    except Exception as e:
        print(f"Error saving embeddings: {e}")

    # Save memories
    try:
        with open(MEMORIES_FILE, "w") as f:
            json.dump(voice_memories, f, indent=2)
    except Exception as e:
        print(f"Error saving memories: {e}")

    # Save chat sessions
    try:
        with open(CHAT_SESSIONS_FILE, "w") as f:
            json.dump(chat_sessions, f, indent=2)
    except Exception as e:
        print(f"Error saving chat sessions: {e}")


@app.route("/")
def dashboard():
    """Main privacy dashboard page"""
    # Load speaker IDs for dropdown
    _, voice_memories, _ = load_databases()
    speaker_ids = list(voice_memories.keys())

    # Generate options for the dropdown
    options_html = ""
    for id in speaker_ids:
        options_html += f'<option value="{id}">{id}</option>'

    # Return HTML page
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Privacy Dashboard - Helbling VoiceBot</title>
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                line-height: 1.6;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                color: #333;
            }}
            h1 {{ color: #2c3e50; }}
            .container {{ 
                background: #f9f9f9; 
                padding: 20px; 
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }}
            label {{ 
                display: block; 
                margin-bottom: 5px;
                font-weight: bold;
            }}
            input, select {{ 
                width: 100%; 
                padding: 8px;
                margin-bottom: 15px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }}
            button {{ 
                background: #3498db; 
                color: white; 
                padding: 10px 15px; 
                border: none; 
                border-radius: 4px;
                cursor: pointer;
            }}
            button:hover {{ 
                background: #2980b9; 
            }}
            button.delete {{ 
                background: #e74c3c; 
            }}
            button.delete:hover {{ 
                background: #c0392b; 
            }}
            .memory-item {{
                background: white;
                padding: 10px;
                margin-bottom: 5px;
                border-radius: 4px;
                border-left: 3px solid #3498db;
            }}
            .hidden {{ display: none; }}
            .warning {{
                background-color: #ffeaa7;
                border-left: 4px solid #fdcb6e;
                padding: 10px;
                margin-bottom: 15px;
            }}
            .header {{
                background-color: #2c3e50;
                color: white;
                padding: 20px;
                text-align: center;
                margin-bottom: 20px;
                border-radius: 8px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Helbling VoiceBot Privacy Dashboard</h1>
            <p>View and manage your stored voice data</p>
        </div>
        
        <div class="container">
            <h2>Access Your Data</h2>
            <p>Enter your Speaker ID or select from known IDs to manage your voice data.</p>
            
            <div>
                <label for="speaker-id">Your Speaker ID:</label>
                <input type="text" id="speaker-id" placeholder="Enter your Speaker ID">
                
                <label for="known-ids">Or select from known IDs:</label>
                <select id="known-ids" onchange="document.getElementById('speaker-id').value = this.value">
                    <option value="">-- Select ID --</option>
                    {options_html}
                </select>
                
                <button onclick="loadUserData()">View My Data</button>
            </div>
        </div>
        
        <div id="user-data" class="container hidden">
            <h2>Your Stored Data</h2>
            <div id="data-summary"></div>
            
            <h3>Your Memories</h3>
            <div id="memories-list"></div>
            
            <div class="warning">
                <strong>Warning:</strong> Deleting your data is permanent and cannot be undone.
            </div>
            
            <div>
                <button class="delete" onclick="deleteUserData()">Delete All My Data</button>
            </div>
        </div>
        
        <script>
            async function loadUserData() {{
                const speakerId = document.getElementById('speaker-id').value;
                if (!speakerId) {{
                    alert('Please enter your Speaker ID');
                    return;
                }}
                
                try {{
                    const response = await fetch(`/api/users/${{speakerId}}`);
                    if (!response.ok) {{
                        throw new Error('User not found');
                    }}
                    
                    const data = await response.json();
                    
                    // Show user data section
                    document.getElementById('user-data').classList.remove('hidden');
                    
                    // Display summary
                    const summary = document.getElementById('data-summary');
                    summary.innerHTML = `
                        <p><strong>Speaker ID:</strong> ${{data.speaker_id}}</p>
                        <p><strong>Voice Profile Stored:</strong> ${{data.has_voice_profile ? 'Yes' : 'No'}}</p>
                        <p><strong>Number of Memories:</strong> ${{data.memories.length}}</p>
                        <p><strong>Associated Chat Sessions:</strong> ${{data.chat_sessions.length}}</p>
                    `;
                    
                    // Display memories
                    const memoriesList = document.getElementById('memories-list');
                    memoriesList.innerHTML = '';
                    
                    if (data.memories.length === 0) {{
                        memoriesList.innerHTML = '<p>No memories stored.</p>';
                    }} else {{
                        data.memories.forEach(memory => {{
                            const memoryText = typeof memory === 'string' ? memory : memory.text;
                            const memoryDiv = document.createElement('div');
                            memoryDiv.className = 'memory-item';
                            memoryDiv.textContent = memoryText;
                            memoriesList.appendChild(memoryDiv);
                        }});
                    }}
                }} catch (error) {{
                    alert('Error: ' + error.message);
                }}
            }}
            
            async function deleteUserData() {{
                const speakerId = document.getElementById('speaker-id').value;
                if (!speakerId) {{
                    alert('Please enter your Speaker ID');
                    return;
                }}
                
                if (confirm('Are you sure you want to delete ALL your data? This cannot be undone.')) {{
                    try {{
                        const response = await fetch(`/api/users/${{speakerId}}`, {{
                            method: 'DELETE'
                        }});
                        
                        if (!response.ok) {{
                            throw new Error('Failed to delete data');
                        }}
                        
                        const result = await response.json();
                        alert('Your data has been deleted successfully');
                        
                        // Reset UI
                        document.getElementById('user-data').classList.add('hidden');
                        document.getElementById('speaker-id').value = '';
                        document.getElementById('known-ids').value = '';
                        
                        // Reload the page to refresh the list of known IDs
                        location.reload();
                    }} catch (error) {{
                        alert('Error: ' + error.message);
                    }}
                }}
            }}
        </script>
    </body>
    </html>
    """


@app.route("/api/users/<speaker_id>", methods=["GET"])
def get_user_data(speaker_id):
    """API endpoint to get user data"""
    voice_embeddings, voice_memories, chat_sessions = load_databases()

    # Check if user exists
    if speaker_id not in voice_memories and speaker_id not in voice_embeddings:
        return jsonify({"error": "User not found"}), 404

    # Find associated chat sessions
    associated_sessions = []
    for session_id, session_data in chat_sessions.items():
        if session_data.get("speaker_id") == speaker_id:
            associated_sessions.append(session_id)

    # Prepare response
    user_data = {
        "speaker_id": speaker_id,
        "has_voice_profile": speaker_id in voice_embeddings,
        "memories": voice_memories.get(speaker_id, []),
        "chat_sessions": associated_sessions,
    }

    return jsonify(user_data)


@app.route("/api/users/<speaker_id>", methods=["DELETE"])
def delete_user_data(speaker_id):
    """API endpoint to delete user data"""
    voice_embeddings, voice_memories, chat_sessions = load_databases()

    # Check if user exists
    if speaker_id not in voice_memories and speaker_id not in voice_embeddings:
        return jsonify({"error": "User not found"}), 404

    # Remove voice profile
    if speaker_id in voice_embeddings:
        del voice_embeddings[speaker_id]

    # Remove memories
    if speaker_id in voice_memories:
        del voice_memories[speaker_id]

    # Update chat sessions
    for session_id, session_data in chat_sessions.items():
        if session_data.get("speaker_id") == speaker_id:
            chat_sessions[session_id]["speaker_id"] = None

    # Save updated databases
    save_databases(voice_embeddings, voice_memories, chat_sessions)

    return jsonify({"status": "success", "message": "User data deleted successfully"})


if __name__ == "__main__":
    print("Starting standalone Privacy Dashboard")
    print("Privacy dashboard available at: http://localhost:5001")
    app.run(debug=True, host="0.0.0.0", port=5001)
