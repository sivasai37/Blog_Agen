from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

class GroqLLM:
    def __init__(self):
        load_dotenv()  # Load environment variables from .env file 

    def get_llm(self):
        try:

          os.environ["GROQ_API_KEY"] =  self.get_groq_api_key =  os.getenv("GROQ_API_KEY")  # Set the GROQ API key from environment variable
          llm = ChatGroq(api_key=self.get_groq_api_key,model="qwen/qwen3.8-27b")
          return llm  # Initialize the ChatGroq LLM
        except Exception as e:
               raise ValueError(f"Error initializing GroqLLM: {e}")  # Raise an error if initialization fails

