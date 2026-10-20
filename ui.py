import requests
import streamlit as st

API_URL = "http://localhost:8000"
STATUS_ICONS = {"processing": "⏳", "ready": "✅", "failed": "❌"}

st.set_page_config(page_title="Document Q&A", page_icon="📄")

# Let each paragraph pick its own direction, so Arabic and English both render correctly
st.markdown(
    """
    <style>
    [data-testid="stChatMessage"] p { unicode-bidi: plaintext; text-align: start; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "documents" not in st.session_state:
    st.session_state.documents = []  # [{"id", "filename", "status", "error"}]
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role", "content", "sources"}]


def error_detail(response: requests.Response) -> str:
    try:
        return response.json().get("detail", response.text)
    except ValueError:
        return response.text


def upload_pdf(file) -> None:
    try:
        response = requests.post(
            f"{API_URL}/documents",
            files={"file": (file.name, file.getvalue(), "application/pdf")},
            timeout=60,
        )
    except requests.RequestException as e:
        st.sidebar.error(f"Could not reach the API: {e}")
        return
    if response.status_code != 200:
        st.sidebar.error(error_detail(response))
        return
    data = response.json()
    st.session_state.documents.append(
        {"id": data["document_id"], "filename": data["filename"], "status": "processing", "error": None}
    )


def refresh_status(doc: dict) -> None:
    if doc["status"] in ("ready", "failed"):
        return  # final states don't change
    try:
        response = requests.get(f"{API_URL}/documents/{doc['id']}/status", timeout=10)
    except requests.RequestException:
        return
    if response.status_code == 200:
        data = response.json()
        doc["status"], doc["error"] = data["status"], data["error"]
    elif response.status_code == 404:
        doc["status"], doc["error"] = "failed", "Unknown to the server (it was probably restarted)"


@st.fragment(run_every=3)
def documents_panel() -> None:
    """Re-runs every 3 seconds on its own, so statuses update without touching the chat."""
    if not st.session_state.documents:
        st.caption("No documents uploaded yet.")
    for doc in st.session_state.documents:
        refresh_status(doc)
        st.write(f"{STATUS_ICONS.get(doc['status'], '•')} {doc['filename']}  \n`{doc['status']}`")
        if doc["error"]:
            st.caption(doc["error"])


def ask(prompt: str, temperature: float) -> dict:
    try:
        response = requests.post(
            f"{API_URL}/generate/text",
            json={"prompt": prompt, "temperature": temperature},
            timeout=120,
        )
    except requests.RequestException as e:
        return {"content": f"Could not reach the API: {e}", "sources": []}
    if response.status_code != 200:
        return {"content": f"Error: {error_detail(response)}", "sources": []}
    return response.json()


def show_sources(sources: list[dict]) -> None:
    if sources:
        with st.expander("Sources"):
            for s in sources:
                st.write(f"{s['source']}, page {s['page']} (score {s['score']:.2f})")


# ---------- Sidebar: upload + document list + settings ----------
with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
    if st.button("Upload", disabled=uploaded is None):
        upload_pdf(uploaded)
    documents_panel()

    st.divider()
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------- Main: chat ----------
st.title("Ask your documents")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_sources(message.get("sources", []))

if prompt := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask(prompt, temperature)
        st.markdown(result["content"])
        show_sources(result["sources"])
    st.session_state.messages.append(
        {"role": "assistant", "content": result["content"], "sources": result["sources"]}
    )
