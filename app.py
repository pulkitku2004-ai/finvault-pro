import streamlit as st
import requests

st.set_page_config(page_title="FinVault AI", layout="wide")

st.title("📊 FinVault AI")
st.caption("Hybrid Graph + Vector Financial Research Assistant")

API_URL = "http://127.0.0.1:8000/query"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_query = st.chat_input("Ask about the financial report...")

if user_query:

    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    # Query backend
    with st.spinner("Analyzing financial documents..."):

        try:
            response = requests.post(
                API_URL,
                json={"query": user_query}
            )

            result = response.json()

            answer = result.get("answer", "No answer returned.")
            documents_used = result.get("documents_used", 0)
            sources = result.get("sources", [])

        except Exception as e:
            answer = f"API Error: {e}"
            documents_used = 0
            sources = []

    # Show assistant message
    with st.chat_message("assistant"):
        st.write(answer)

        st.markdown(f"📄 **Documents used:** {documents_used}")

        if sources:
            st.markdown("### Sources")

            for i, src in enumerate(sources):
                st.write(f"**Doc {i+1}:** {src[:300]}...")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })