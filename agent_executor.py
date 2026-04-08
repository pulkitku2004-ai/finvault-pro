from agent_router import route_question
from tools import vector_search_tool, graph_search_tool, calculator_tool
import time
from llm_config import llm

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