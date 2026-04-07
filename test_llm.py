#from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
#from langchain_openai import ChatOpenAI
# 1. Load your .env file
load_dotenv()

# 2. Configure the Groq Judge (Llama 3.3 70B is elite for financial RAG)
llm = ChatGroq(
    model="llama-3.3-70b-versatile", # Aligned Groq Model ID
    api_key=os.getenv("GROQ_API_KEY"), # Ensure this is in your .env
    temperature=0, # Set to 0 for financial accuracy (no "creative" numbers)
    max_tokens=1024, # Increased slightly to allow for detailed risk explanations
    n=1,
    model_kwargs={
        "top_p": 1,
        #stop
    }
)
response = llm.invoke("Explain financial risk in banking.")

print(response.content)