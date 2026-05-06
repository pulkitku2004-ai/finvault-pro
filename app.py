import streamlit as st
import requests

st.set_page_config(page_title="FinVault AI", layout="wide")

st.title("📊 FinVault AI")
st.caption("Hybrid Graph + Vector Financial Research Assistant")

API_URL = "http://127.0.0.1:8000/query"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_query = st.chat_input("Ask about the financial report...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    with st.spinner("Analyzing financial documents..."):
        try:
            response = requests.post(API_URL, json={"query": user_query})
            result = response.json()
        except Exception as e:
            result = {
                "answer": f"API Error: {e}",
                "status": "error",
                "documents_used": 0,
                "sources": [],
                "verification": {},
            }

    answer = result.get("answer") or "No answer returned."
    documents_used = result.get("documents_used", 0)
    sources = result.get("sources", [])
    verification = result.get("verification", {})
    astr_status = result.get("status", "")

    with st.chat_message("assistant"):
        st.write(answer)

        # ASTR-O verification badge
        badge = verification.get("badge", "")
        confidence = verification.get("confidence", "")
        summary = verification.get("summary", "")

        if astr_status == "verified":
            st.success(f"{badge}  |  {summary}")
        elif astr_status in ("unverified", "partial"):
            st.warning(f"{badge}  |  {summary}")
            warning_text = verification.get("warning", "")
            if warning_text:
                st.caption(f"⚠ {warning_text}")
        elif astr_status == "error":
            st.error(f"Audit error: {verification.get('summary', 'Unknown')}")

        st.markdown(f"📄 **Documents used:** {documents_used}  |  Confidence: **{confidence or 'N/A'}**")

        # Source citations with tier badges
        if sources and isinstance(sources[0], dict):
            with st.expander("Sources"):
                for src in sources:
                    tier_badge = src.get("tier_badge", "⚪")
                    doc_name = src.get("document", "Unknown")
                    auth = src.get("authorization", {})
                    auth_badge = auth.get("badge", "")
                    snippet = src.get("snippet", "")
                    st.markdown(
                        f"{tier_badge} **{doc_name}** {auth_badge}  \n_{snippet}_"
                    )
        elif sources:
            with st.expander("Sources"):
                for i, src in enumerate(sources[:3]):
                    st.write(f"**Doc {i+1}:** {str(src)[:300]}...")

    st.session_state.messages.append({"role": "assistant", "content": answer})
