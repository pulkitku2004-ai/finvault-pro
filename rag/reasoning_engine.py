from llm_config import llm
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
    
