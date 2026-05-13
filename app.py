import os
import io
import textwrap
from typing import List, Tuple

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np
import streamlit as st
import fitz  # PyMuPDF
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq


# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(page_title="NEURON · RAG Intelligence", page_icon="🧠", layout="centered")


# -----------------------------
# Custom CSS: Full UI Redesign
# -----------------------------
st.markdown("""
<style>
/* ===== GOOGLE FONTS ===== */
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Mono:wght@300;400;500&family=Outfit:wght@200;300;400;500;600;700&display=swap');

/* ===== ROOT VARIABLES ===== */
:root {
    --bg-primary: #0a0a0f;
    --bg-secondary: #151520;
    --bg-card: rgba(22, 22, 34, 0.9);
    --accent-warm: #e8a642;
    --accent-glow: #f0c060;
    --accent-cool: #6e8cff;
    --accent-rose: #ff6b8a;
    --text-primary: #e8e6e1;
    --text-secondary: #a8a4b4;
    --text-muted: #7a7590;
    --placeholder: rgba(168, 164, 180, 0.5);
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-accent: rgba(232, 166, 66, 0.25);
    --glass-bg: rgba(16, 16, 24, 0.72);
    --glass-border: rgba(255, 255, 255, 0.08);
    --shadow-deep: 0 24px 80px rgba(0, 0, 0, 0.6);
    --shadow-glow: 0 0 40px rgba(232, 166, 66, 0.08);
    --radius-lg: 16px;
    --radius-md: 10px;
    --radius-sm: 6px;
}

/* ===== GLOBAL RESETS ===== */
html, body, .stApp {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Outfit', sans-serif !important;
}

/* Nuclear dark override — kill ALL default white backgrounds */
.stApp div, .stApp section, .stApp article {
    color: inherit;
}

/* Streamlit's internal BaseWeb components (inputs, dropzones, etc.) */
[data-baseweb="input"],
[data-baseweb="base-input"],
[data-baseweb="textarea"],
[data-baseweb="select"],
[data-baseweb="popover"],
[data-testid="stFileUploadDropzone"],
[data-testid="stFileUploadDropzone"] *,
.stChatInput > div,
.stChatInput textarea,
.stChatInput [data-baseweb] {
    background: var(--bg-secondary) !important;
    background-color: var(--bg-secondary) !important;
    color: var(--text-primary) !important;
}

.stApp {
    background: 
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(110, 140, 255, 0.04) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 90%, rgba(232, 166, 66, 0.03) 0%, transparent 60%),
        var(--bg-primary) !important;
}

/* ===== HIDE DEFAULT STREAMLIT CHROME ===== */
#MainMenu, footer, header, .stDeployButton {
    display: none !important;
}

/* Hide default padding noise */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
    max-width: 800px !important;
}

/* ===== GLOBAL DARK OVERRIDES FOR ALL INPUTS ===== */
/* Kill ALL borders on every nested wrapper — only the outermost gets a border */
input, textarea, select,
[data-baseweb] input,
[data-baseweb] textarea,
[data-baseweb="input"],
[data-baseweb="base-input"] {
    background-color: var(--bg-secondary) !important;
    color: var(--text-primary) !important;
    caret-color: var(--accent-warm) !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}

/* ALL placeholder text — make visible */
input::placeholder,
textarea::placeholder,
[data-baseweb] input::placeholder,
[data-baseweb] textarea::placeholder {
    color: var(--placeholder) !important;
    opacity: 1 !important;
    -webkit-text-fill-color: var(--placeholder) !important;
}

/* Hide Streamlit's duplicate inner placeholder div that overlaps */
.stTextInput [data-baseweb="input"] > div:last-child,
.stTextInput [data-baseweb="base-input"] > div:last-child {
    color: var(--placeholder) !important;
    opacity: 0.6 !important;
}

/* === TEXT INPUT: single clean outer border only === */
/* The outermost wrapper gets the styled border */
.stTextInput > div {
    background-color: var(--bg-secondary) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}

.stTextInput > div:focus-within {
    border-color: var(--accent-warm) !important;
    box-shadow: 0 0 20px rgba(232, 166, 66, 0.1) !important;
}

/* Strip borders from ALL inner wrappers */
.stTextInput > div > div,
.stTextInput > div > div > div,
.stTextInput [data-baseweb="input"],
.stTextInput [data-baseweb="base-input"],
.stTextInput input {
    background-color: var(--bg-secondary) !important;
    color: var(--text-primary) !important;
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    outline: none !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.95rem !important;
}

.stTextInput label {
    color: var(--text-secondary) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.88rem !important;
}

/* Password eye icon */
.stTextInput button {
    color: var(--text-muted) !important;
    border: none !important;
    background: transparent !important;
}
.stTextInput button:hover {
    color: var(--accent-warm) !important;
}

/* Kill the InputInstructions ghost border */
[data-testid="InputInstructions"],
[data-testid="InputInstructions"] div {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    background: transparent !important;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #0d0d14 0%, #0f0f1a 40%, #121220 100%) !important;
    border-right: 1px solid var(--border-subtle) !important;
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Instrument Serif', serif !important;
    color: var(--accent-warm) !important;
    letter-spacing: -0.02em;
}

section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.88rem !important;
    color: var(--text-secondary) !important;
    line-height: 1.7 !important;
}

/* Sidebar input — inherit the global single-border fix, no extra rules needed */
section[data-testid="stSidebar"] .stTextInput > div > div,
section[data-testid="stSidebar"] .stTextInput > div > div > div,
section[data-testid="stSidebar"] .stTextInput [data-baseweb="input"],
section[data-testid="stSidebar"] .stTextInput [data-baseweb="base-input"],
section[data-testid="stSidebar"] .stTextInput input {
    background: var(--bg-secondary) !important;
    background-color: var(--bg-secondary) !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    color: var(--text-primary) !important;
    font-family: 'DM Mono', monospace !important;
}

section[data-testid="stSidebar"] .stAlert {
    background: rgba(232, 166, 66, 0.08) !important;
    border: 1px solid rgba(232, 166, 66, 0.2) !important;
    border-radius: var(--radius-md) !important;
    color: var(--accent-glow) !important;
    font-family: 'DM Mono', monospace !important;
}

/* ===== MAIN TITLE AREA ===== */
.main-hero {
    text-align: center;
    padding: 0.5rem 0 1rem;
    position: relative;
}

.main-hero::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(232, 166, 66, 0.06) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

.main-hero .brand-icon {
    font-size: 2.2rem;
    margin-bottom: 0.3rem;
    display: block;
    filter: drop-shadow(0 0 20px rgba(232, 166, 66, 0.3));
    animation: float 4s ease-in-out infinite;
}

@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-6px); }
}

.main-hero h1 {
    font-family: 'Instrument Serif', serif !important;
    font-size: 2.8rem;
    font-weight: 400;
    color: var(--text-primary);
    letter-spacing: -0.03em;
    margin: 0;
    line-height: 1.1;
    position: relative;
    z-index: 1;
}

.main-hero h1 span {
    background: linear-gradient(135deg, var(--accent-warm), var(--accent-glow), var(--accent-rose));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.main-hero .tagline {
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    font-weight: 400;
    color: var(--text-secondary);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.6rem;
    position: relative;
    z-index: 1;
}

/* ===== FILE UPLOADER ===== */
.stFileUploader,
[data-testid="stFileUploader"],
[data-testid="stFileUploadDropzone"] {
    background: var(--bg-secondary) !important;
    background-color: var(--bg-secondary) !important;
    border: 1px dashed rgba(232, 166, 66, 0.15) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1rem !important;
    transition: all 0.4s ease;
}

/* Kill the light grey inline background on the dropzone */
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploaderDropzone"] > section,
[data-testid="stFileUploaderDropzone"] > div,
[data-testid="stFileUploaderDropzone"] section {
    background: var(--bg-secondary) !important;
    background-color: var(--bg-secondary) !important;
    border: none !important;
    box-shadow: none !important;
}

/* The inner drop zone section */
[data-testid="stFileUploadDropzone"] > div,
[data-testid="stFileUploadDropzone"] section,
.stFileUploader section {
    background: var(--bg-secondary) !important;
    background-color: var(--bg-secondary) !important;
    color: var(--text-secondary) !important;
    padding: 0.8rem !important;
}

/* "Drag and drop" text */
[data-testid="stFileUploadDropzone"] span,
[data-testid="stFileUploadDropzone"] small,
.stFileUploader span {
    color: var(--text-primary) !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 400 !important;
    font-size: 0.95rem !important;
}

/* File size limit subtitle */
[data-testid="stFileUploadDropzone"] small {
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
}

.stFileUploader:hover,
[data-testid="stFileUploadDropzone"]:hover {
    border-color: rgba(232, 166, 66, 0.4) !important;
    box-shadow: var(--shadow-glow);
}

.stFileUploader label {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 400 !important;
    color: var(--text-secondary) !important;
    font-size: 0.95rem !important;
}

/* Browse files button */
.stFileUploader button,
[data-testid="stFileUploadDropzone"] button,
[data-testid="baseButton-secondary"] {
    background: linear-gradient(135deg, var(--accent-warm), #d4943a) !important;
    color: #0a0a0f !important;
    font-family: 'DM Mono', monospace !important;
    font-weight: 500 !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.05em;
    transition: all 0.3s ease;
}

.stFileUploader button:hover,
[data-testid="stFileUploadDropzone"] button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 24px rgba(232, 166, 66, 0.25) !important;
}

/* File size limit text */
[data-testid="stFileUploader"] small,
.uploadedFileName {
    color: var(--text-secondary) !important;
    font-family: 'DM Mono', monospace !important;
}

/* ===== CHAT MESSAGES ===== */
.stChatMessage {
    background: transparent !important;
    border: none !important;
    padding: 0.8rem 0 !important;
}

/* User messages */
.stChatMessage[data-testid="stChatMessage"]:has(.stChatMessageAvatarUser),
div[data-testid="stChatMessage"]:nth-child(odd) {
    background: transparent !important;
}

.stChatMessage .stMarkdown p {
    font-family: 'Outfit', sans-serif !important;
    font-size: 1rem !important;
    line-height: 1.75 !important;
    color: var(--text-primary) !important;
}

/* Chat message container styling */
[data-testid="stChatMessageContent"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1rem 1.3rem !important;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
}

/* User message accent */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
    border-left: 2px solid var(--accent-warm) !important;
    background: rgba(232, 166, 66, 0.03) !important;
}

/* Assistant message accent */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {
    border-left: 2px solid var(--accent-cool) !important;
    background: rgba(110, 140, 255, 0.03) !important;
}

/* Chat avatars */
.stChatMessage img,
[data-testid="chatAvatarIcon-user"],
[data-testid="chatAvatarIcon-assistant"] {
    border-radius: 50% !important;
    border: 1px solid var(--border-subtle) !important;
}

/* ===== CHAT INPUT ===== */
.stChatInput,
.stChatInput > div,
.stChatInput > div > div,
.stChatInput > div > div > div,
.stChatInput > div > div > div > div,
.stChatInput div,
.stChatInput form,
.stChatInput [data-baseweb],
.stChatInput [data-baseweb] > div,
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div > div,
[data-testid="stChatInput"] div,
[data-testid="stChatInput"] form,
[data-testid="stChatInput"] [data-baseweb],
[data-testid="stChatInput"] [data-baseweb] > div,
[data-testid="stBottom"] > div,
[data-testid="stBottom"] > div > div {
    border: none !important;
    border-top: none !important;
    border-bottom: none !important;
    border-left: none !important;
    border-right: none !important;
    border-width: 0 !important;
    box-shadow: none !important;
    outline: none !important;
    background-color: transparent !important;
    background: transparent !important;
}

/* Re-apply only the outermost visual container */
[data-testid="stChatInput"] > div:first-child {
    background: var(--bg-secondary) !important;
    background-color: var(--bg-secondary) !important;
    border-radius: var(--radius-lg) !important;
    padding: 0 !important;
    overflow: hidden;
}

/* Kill Streamlit emotion-cache wrappers creating ghost boxes */
[class*="st-emotion-cache"][class*="stChatInput"] div,
.stChatInput [class*="st-emotion-cache"] {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
}

.stChatInput textarea,
[data-testid="stChatInput"] textarea {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 400 !important;
    font-size: 1rem !important;
    color: var(--text-primary) !important;
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    border-width: 0 !important;
    box-shadow: none !important;
    outline: none !important;
}

.stChatInput textarea::placeholder,
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-secondary) !important;
    opacity: 1 !important;
    -webkit-text-fill-color: var(--text-secondary) !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 400 !important;
    font-size: 0.95rem !important;
}

/* Chat submit button */
.stChatInput button {
    background: linear-gradient(135deg, var(--accent-warm), #d4943a) !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    transition: all 0.3s ease;
}

.stChatInput button:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 16px rgba(232, 166, 66, 0.3) !important;
}

.stChatInput button svg {
    fill: #0a0a0f !important;
}

/* ===== SPINNER ===== */
.stSpinner > div {
    border-color: var(--accent-warm) transparent transparent transparent !important;
}

.stSpinner > div > span {
    font-family: 'DM Mono', monospace !important;
    color: var(--text-secondary) !important;
    font-size: 0.8rem !important;
}

/* ===== WARNINGS ===== */
.stAlert {
    background: rgba(255, 107, 138, 0.06) !important;
    border: 1px solid rgba(255, 107, 138, 0.15) !important;
    border-radius: var(--radius-md) !important;
    color: var(--accent-rose) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
}

/* ===== DIVIDER ===== */
hr {
    border: none !important;
    border-top: 1px solid var(--border-subtle) !important;
    margin: 2rem 0 1rem !important;
}

/* ===== CAPTION / FOOTER ===== */
.footer-section {
    text-align: center;
    padding: 1rem 0 0.5rem;
}

.footer-section p {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    color: var(--text-secondary) !important;
    letter-spacing: 0.08em;
}

.footer-section .dot {
    display: inline-block;
    width: 4px;
    height: 4px;
    background: var(--accent-warm);
    border-radius: 50%;
    margin: 0 8px;
    vertical-align: middle;
    animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 0.3; }
    50% { opacity: 1; }
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar {
    width: 6px;
}

::-webkit-scrollbar-track {
    background: var(--bg-primary);
}

::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.15);
}

/* ===== STATUS PILLS ===== */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.04em;
}

.status-pill.ready {
    background: rgba(100, 220, 130, 0.08);
    border: 1px solid rgba(100, 220, 130, 0.2);
    color: #64dc82;
}

.status-pill.waiting {
    background: rgba(232, 166, 66, 0.08);
    border: 1px solid rgba(232, 166, 66, 0.2);
    color: var(--accent-warm);
}

.status-dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    animation: pulse 2s ease-in-out infinite;
}

.status-dot.ready { background: #64dc82; }
.status-dot.waiting { background: var(--accent-warm); }

/* ===== UPLOAD ZONE ENHANCEMENT ===== */
.upload-zone {
    text-align: center;
    padding: 1rem 0 0.5rem;
}

.upload-zone .upload-label {
    font-family: 'Instrument Serif', serif;
    font-size: 1.2rem;
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
}

.upload-zone .upload-hint {
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    color: var(--text-secondary);
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Custom Hero Header
# -----------------------------
st.markdown("""
<div class="main-hero">
    <span class="brand-icon">🧠</span>
    <h1>NEURON <span>RAG</span></h1>
    <p class="tagline">retrieval-augmented intelligence · groq + faiss</p>
</div>
""", unsafe_allow_html=True)


# -----------------------------
# Sidebar: instructions and status
# -----------------------------
with st.sidebar:
    st.markdown("### ⚙ Configuration")

    api_key_input = st.text_input(
        "Groq API key",
        type="password",
        help="You can also set the GROQ_API_KEY environment variable.",
    )

    st.markdown("---")
    st.markdown("### 📐 Architecture")
    st.markdown(
        "- **Chunking** — 500 char windows, 80 char overlap\n"
        "- **Embeddings** — `all-MiniLM-L6-v2` transformer\n"
        "- **Index** — FAISS L2 nearest-neighbor\n"
        "- **LLM** — Groq `llama-3.1-8b-instant`"
    )


# -----------------------------
# Utility functions (UNCHANGED)
# -----------------------------

def read_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file."""
    text = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text.append(page.get_text())
    return "\n".join(text).strip()


def read_txt(file_bytes: bytes) -> str:
    """Read text from a TXT file."""
    return file_bytes.decode("utf-8", errors="ignore").strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    """Split text into overlapping chunks."""
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Create a FAISS index from embeddings."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    """Generate embeddings for a list of texts."""
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return vectors.astype("float32")


def retrieve_context(
    query: str,
    model: SentenceTransformer,
    index: faiss.IndexFlatL2,
    chunks: List[str],
    top_k: int = 4,
) -> Tuple[str, List[str]]:
    """Retrieve top-k relevant chunks for the query."""
    query_vector = embed_texts(model, [query])
    distances, indices = index.search(query_vector, top_k)

    results = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            results.append(chunks[idx])

    context = "\n\n".join(results)
    return context, results


def format_prompt(context: str, question: str) -> str:
    """Create a simple RAG prompt with context and question."""
    prompt = f"""
You are a helpful assistant. Use the context below to answer the question.
If the answer is not in the context, say you do not know.

Context:
{context}

Question:
{question}

Answer:
"""
    return textwrap.dedent(prompt).strip()


def call_groq(client: Groq, prompt: str) -> str:
    """Call Groq LLM and return the response text."""
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        if response.choices:
            return response.choices[0].message.content.strip()
        return "I could not generate a response."
    except Exception as exc:
        return f"Error calling Groq API: {exc}"


# -----------------------------
# Session state (UNCHANGED)
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "model" not in st.session_state:
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        st.session_state.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    except Exception:
        st.session_state.model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# Upload Zone
# -----------------------------
st.markdown("""
<div class="upload-zone">
    <p class="upload-label">Feed the knowledge base</p>
    <p class="upload-hint">PDF or TXT — your document becomes searchable context</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"], label_visibility="collapsed")

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    if not file_bytes:
        st.warning("The uploaded file is empty.")
    else:
        if uploaded_file.name.lower().endswith(".pdf"):
            raw_text = read_pdf(file_bytes)
        else:
            raw_text = read_txt(file_bytes)

        if not raw_text:
            st.warning("No text could be extracted from the file.")
        else:
            chunks = chunk_text(raw_text)
            embeddings = embed_texts(st.session_state.model, chunks)
            index = build_faiss_index(embeddings)

            st.session_state.chunks = chunks
            st.session_state.index = index

            with st.sidebar:
                st.success(f"✦ Indexed {len(chunks)} chunks from `{uploaded_file.name}`")

# Status indicator
if st.session_state.index is not None and st.session_state.chunks:
    st.markdown(f"""
    <div style="text-align: center; margin: 0.5rem 0 1.5rem;">
        <span class="status-pill ready">
            <span class="status-dot ready"></span>
            knowledge base active · {len(st.session_state.chunks)} chunks indexed
        </span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="text-align: center; margin: 0.5rem 0 1.5rem;">
        <span class="status-pill waiting">
            <span class="status-dot waiting"></span>
            awaiting document upload
        </span>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------
# Chat interface (UNCHANGED logic)
# -----------------------------
for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(content)

user_question = st.chat_input("Ask something about your document...")

if user_question:
    if st.session_state.index is None or not st.session_state.chunks:
        st.warning("Please upload a document first.")
    else:
        groq_key = api_key_input or os.getenv("GROQ_API_KEY")
        if not groq_key:
            st.warning("Please provide your Groq API key in the sidebar.")
        else:
            st.session_state.chat_history.append(("user", user_question))
            with st.chat_message("user"):
                st.write(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving context & generating..."):
                    context, _ = retrieve_context(
                        user_question,
                        st.session_state.model,
                        st.session_state.index,
                        st.session_state.chunks,
                    )

                    prompt = format_prompt(context, user_question)
                    client = Groq(api_key=groq_key)
                    answer = call_groq(client, prompt)

                st.write(answer)
                st.session_state.chat_history.append(("assistant", answer))


# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("""
<div class="footer-section">
    <p>NEURON RAG <span class="dot"></span> Groq + FAISS + SentenceTransformers <span class="dot"></span> Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
