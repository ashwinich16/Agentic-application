import streamlit as st
import requests

API_URL = "http://localhost:8000/process"

if "session_id" not in st.session_state:
    st.session_state["session_id"] = None

st.title("Agentic Application UI")

user_text = st.text_area("Enter your text")

uploaded_files = st.file_uploader(
    "Upload files (images / PDF / audio)",
    accept_multiple_files=True,
    type=["png", "jpg", "jpeg", "pdf", "mp3", "wav", "m4a"],
)

if st.button("Send"):
    uploaded_files = uploaded_files or []  

    data = {"text": user_text or ""}
    files = [
        ("files", (f.name, f.getvalue(), f.type))
        for f in uploaded_files
    ]

    headers = {}
    if st.session_state["session_id"]:
        headers["session_id"] = st.session_state["session_id"]

    with st.spinner("Processing..."):
        response = requests.post(
            API_URL,
            data=data,
            files=files,
            headers=headers,
            timeout=120,
        )

    if response.status_code != 200:
        st.error(f"Request failed: {response.status_code}")
        st.stop()

    result = response.json()

    # save session id
    if result.get("session_id"):
        st.session_state["session_id"] = result["session_id"]

    resp = result.get("response", {})
    status = resp.get("status")

    if status == "follow_up":
        st.warning(resp.get("question", "Need more details."))
    elif status == "completed":
        st.success("Task Completed")
        st.write(resp.get("result", ""))
    else:
        st.error("Unexpected response format")
