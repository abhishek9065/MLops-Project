import os

import requests
import streamlit as st


API_BASE_URL = st.sidebar.text_input("API URL", value=os.getenv("API_BASE_URL", "http://localhost:8000"))

st.title("Enterprise RAGOps Platform")
st.caption("Phases 2-4: RAG answers, citations, feedback, and trace inspection.")

upload_tab, ask_tab, trace_tab = st.tabs(["Upload", "Ask", "Trace"])

with upload_tab:
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

with ask_tab:
    question = st.text_input("Ask a question")
    top_k = st.slider("Retrieved chunks", min_value=1, max_value=10, value=4)
    prompt_version = st.selectbox("Prompt version", options=["v1", "v2"], index=0)
    if question and st.button("Ask"):
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json={"question": question, "top_k": top_k, "prompt_version": prompt_version},
            timeout=90,
        )
        if response.ok:
            payload = response.json()
            st.session_state["last_trace_id"] = payload["trace_id"]
            st.subheader("Answer")
            st.write(payload["answer"])
            metric_cols = st.columns(5)
            metric_cols[0].metric("Confidence", payload["confidence"])
            metric_cols[1].metric("Total latency", f'{payload["total_latency_ms"]} ms')
            metric_cols[2].metric("LLM latency", f'{payload["llm_latency_ms"]} ms')
            metric_cols[3].metric("Model", payload["model"])
            metric_cols[4].metric("Cost", f'${payload["estimated_cost_usd"]:.6f}')
            st.caption(f'Trace ID: {payload["trace_id"]}')

            feedback_cols = st.columns(2)
            if feedback_cols[0].button("Thumbs up"):
                feedback_response = requests.post(
                    f'{API_BASE_URL}/traces/{payload["trace_id"]}/feedback',
                    json={"feedback": "up"},
                    timeout=30,
                )
                st.success(feedback_response.json()["message"] if feedback_response.ok else feedback_response.text)
            if feedback_cols[1].button("Thumbs down"):
                feedback_response = requests.post(
                    f'{API_BASE_URL}/traces/{payload["trace_id"]}/feedback',
                    json={"feedback": "down"},
                    timeout=30,
                )
                st.warning(feedback_response.json()["message"] if feedback_response.ok else feedback_response.text)

            st.subheader("Citations")
            for citation in payload["citations"]:
                with st.expander(f'{citation["filename"]} - chunk {citation["chunk_index"]}'):
                    st.write(citation["text"])
                    st.caption(f'Score: {citation["score"]} | Chunk ID: {citation["chunk_id"]}')
        else:
            st.error(response.text)

with trace_tab:
    default_trace_id = st.session_state.get("last_trace_id", "")
    trace_id = st.text_input("Trace ID", value=default_trace_id)
    if trace_id and st.button("Load trace"):
        response = requests.get(f"{API_BASE_URL}/traces/{trace_id}", timeout=30)
        if response.ok:
            trace = response.json()
            st.subheader("Trace")
            st.json(
                {
                    "trace_id": trace["trace_id"],
                    "question": trace["question"],
                    "model": trace["llm_model"],
                    "prompt_version": trace["prompt_version"],
                    "feedback_score": trace["feedback_score"],
                    "latency_ms": trace["total_latency_ms"],
                    "estimated_cost_usd": trace["estimated_cost_usd"],
                }
            )
            with st.expander("Prompt"):
                st.code(trace["prompt"])
            with st.expander("Answer"):
                st.write(trace["answer"])
            with st.expander("Retrieved chunks"):
                st.json(trace["citations"])
        else:
            st.error(response.text)
