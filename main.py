import os
from pathlib import Path
from sarvamai import SarvamAI

# Load .env file if present
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    with open(env_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip().strip("'\""))

api_key = os.environ.get("SARVAM_API_KEY")
if not api_key:
    raise ValueError("SARVAM_API_KEY is not set. Please add your key to the .env file.")

# Initialize the Sarvam client with the key from environment
client = SarvamAI(api_subscription_key=api_key)

# Make a chat completion call
response = client.chat.completions(
    model="sarvam-105b-conversations",
    messages=[
        {"role": "user", "content": "Hello! Please give a brief greeting."}
    ],
)

# Print the reply
print(response.choices[0].message.content)
