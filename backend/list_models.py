from dotenv import load_dotenv
from openai import OpenAI
import os

# Load .env file
load_dotenv()

api_key = os.getenv('GROQ_API_KEY')
print(f"API Key found: {api_key[:20]}..." if api_key else "ERROR: No API key in .env!")

if not api_key:
    print("\nCheck your .env file has:")
    print("GROQ_API_KEY=gsk_xxxxx")
    exit()

client = OpenAI(
    api_key=api_key,
    base_url='https://api.groq.com/openai/v1'
)

print("\nAvailable Groq Models:")
print("=" * 50)

try:
    models = client.models.list()
    for model in models.data:
        print(model.id)
except Exception as e:
    print(f"Error: {e}")