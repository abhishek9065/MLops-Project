import requests
import streamlit as st


API_BASE_URL = st.sidebar.text_input("API URL", value="http://localhost:8000")

st.title("Enterprise RAGOps Platform")
st.caption("Phase 1: document upload, chunking, local embeddings, and retrieval inspection.")

uploaded_file = st.file_uploader("Upload PDF, TXT, or Markdown", type=["pdf", "txt", "md", "markdown"])
if uploaded_file and st.button("Upload document"):
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    response = requests.post(f"{API_BASE_URL}/documents/upload", files=files, timeout=60)
    if response.ok:
        payload = response.json()
        st.success(payload["message"])
        st.json(payload["document"])
    else:
        st.error(response.text)

question = st.text_input("Ask a question")
if question and st.button("Retrieve context"):
    response = requests.post(f"{API_BASE_URL}/ask", json={"question": question, "top_k": 4}, timeout=60)
    if response.ok:
        payload = response.json()
        st.write(payload["answer"])
        st.metric("Confidence", payload["confidence"])
        for citation in payload["citations"]:
            with st.expander(f'{citation["filename"]} - chunk {citation["chunk_index"]}'):
                st.write(citation["text"])
                st.caption(f'Score: {citation["score"]}')
    else:
        st.error(response.text)

