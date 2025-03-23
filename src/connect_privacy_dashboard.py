"""
Launcher for the standalone Privacy Dashboard

This script simply launches the standalone privacy dashboard application,
which operates independently from the main relay server.

Usage:
  python connect_privacy_dashboard.py

This will start the privacy dashboard on port 5001
"""

import sys
import os

# Print current working directory for debugging
current_dir = os.getcwd()
print(f"Current working directory: {current_dir}")

# Check if we're running from src directory
if os.path.basename(current_dir) == "src":
    # We're in src dir, check for voice_database one level up
    expected_db_path = os.path.join(os.path.dirname(current_dir), "voice_database")
else:
    # We're in project root, check for voice_database in current dir
    expected_db_path = os.path.join(current_dir, "voice_database")

print(f"Expected database path: {expected_db_path}")
print(f"Database exists: {os.path.exists(expected_db_path)}")

if os.path.exists(expected_db_path):
    memories_file = os.path.join(expected_db_path, "memories.json")
    print(f"Memories file exists: {os.path.exists(memories_file)}")
    if os.path.exists(memories_file):
        try:
            import json

            with open(memories_file, "r") as f:
                memories = json.load(f)
                print(
                    f"Found {len(memories)} speakers in memories file: {list(memories.keys())}"
                )
        except Exception as e:
            print(f"Error reading memories file: {e}")

# Ensure we're in the src directory
if not os.path.exists("privacy_dashboard.py"):
    print(
        "Error: This script must be run from the src directory containing privacy_dashboard.py"
    )
    sys.exit(1)

# Simply import and run the privacy dashboard
try:
    import privacy_dashboard

    print("Successfully imported privacy dashboard module")
    print("Starting standalone Privacy Dashboard")
    print("Privacy dashboard available at: http://localhost:5001")
    privacy_dashboard.app.run(debug=True, host="0.0.0.0", port=5001)
except ImportError:
    print(
        "Error: Could not import privacy_dashboard.py. Make sure it exists in the same directory."
    )
    sys.exit(1)
except Exception as e:
    print(f"Error launching privacy dashboard: {e}")
    sys.exit(1)
