import streamlit as st
import requests

API_URL = "http://localhost:8000/process"

st.set_page_config(page_title="Agentic Chatbot")

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "messages" not in st.session_state:

    st.session_state.messages = []

st.title("Agentic Application Chatbot")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

with st.sidebar:
    st.header("Attachments (optional)")
    uploaded_files = st.file_uploader(
        "Upload images / PDF / audio",
        accept_multiple_files=True,
        type=["png", "jpg", "jpeg", "pdf", "mp3", "wav", "m4a","ogg"],
    )
    st.caption("Tip: Upload here, then send a message.")

prompt = st.chat_input("Type your message...")
if prompt:
 
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    data = {"text": prompt}
    files_payload = []
    for f in (uploaded_files or []):
        files_payload.append(("files", (f.name, f.getvalue(), f.type)))

    headers = {}
    if st.session_state.session_id:
        headers["session_id"] = st.session_state.session_id

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                r = requests.post(
                    API_URL,
                    data=data,
                    files=files_payload,
                    headers=headers,
                    timeout=120,
                )
            except requests.RequestException as e:
                msg = f"Request error: {e}"
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.stop()

            if r.status_code != 200:
                msg = f"Request failed: {r.status_code}\n\n{r.text}"
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.stop()

            result = r.json()
            if result.get("session_id"):
                st.session_state.session_id = result["session_id"]

            resp = result.get("response", {})
            status = resp.get("status")

            if status == "follow_up":
                assistant_text = resp.get("question", "Need more details.")
            elif status == "completed":
                assistant_text = resp.get("result", "")
            else:
                assistant_text = f"Unexpected response format: {result}"

            st.markdown(assistant_text)
            st.session_state.messages.append({"role": "assistant", "content": assistant_text})
