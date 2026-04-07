#from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os
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
    }
)
def verify_answer(question, context, answer):
    prompt = f"""
    [SYSTEM]
    You are a strict financial verification bot. 
    Your ONLY job is to output the final, verified answer. 
    Do NOT include your reasoning, your thoughts, or the word "Corrected answer:".

    CONTEXT: {context}
    QUESTION: {question}
    DRAFT: {answer}

    [INSTRUCTIONS]
    1. If the draft is supported by context, return the draft exactly.
    2. If the draft has errors, fix them and return ONLY the corrected text.
    3. If there is NO context, return "Data not available."

    VERIFIED OUTPUT:
    """
    response = llm.invoke(prompt)
    return response.content.strip()
    
