from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

# Load API token from .env
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize HuggingFaceEndpoint
model = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    huggingfacehub_api_token=hf_token
)

# Use a list of strings for prompts
prompts = [
    "What is the capital of India?",
    "Who wrote Hamlet?"
]
response = model.generate(prompts)

# Access the generated text
for i, prompt in enumerate(prompts):
    print(f"Prompt: {prompt}")
    print(f"Response: {response.generations[i][0].text}\n")