from agent_router import route_question
from tools import vector_search_tool, graph_search_tool, calculator_tool
from langchain_ollama import ChatOllama
import time
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

def agent_query(question):

    start = time.time()

    tool = route_question(question)

    print("Selected Tool:", tool)

    if tool == "vector_search":

        context = vector_search_tool(question)

    elif tool == "graph_search":

        context = graph_search_tool(question)

    elif tool == "calculator":

        context = calculator_tool(question)

    else:

        context = "No tool available."

    prompt = f"""
        Answer the question using the following context.

        Context:
            {context}

        Question:
            {question}

        Provide a clear financial explanation.
            """
    latency = time.time() - start
    print("Latency:", latency)

    response = llm.invoke(prompt)

    return response.content


if __name__ == "__main__":

    question = "Who spoke during HDFC Q3 earnings?"

    answer = agent_query(question)

    print("\nAnswer:\n", answer)