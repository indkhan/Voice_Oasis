import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
URL = "https://api.openai.com/v1/chat/completions"

headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

data = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
}

response = requests.post(URL, headers=headers, json=data)

print(response.json())
