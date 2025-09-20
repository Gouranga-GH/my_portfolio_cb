import os
from pathlib import Path
from typing import List, Dict

import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS, DocArrayInMemorySearch

import docx2txt

# Removed guardrails import due to Python 3.12 compatibility issues

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage


APP_TITLE = "Hi, I am Gouranga Jha — ask me a question"
PORTFOLIO_URL = "https://gouranga-gh.github.io/my-portfolio/"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESUME_PATH = DATA_DIR / "Gouranga_Resume.docx"
SUMMARY_PATH = DATA_DIR / "Projects_Summary.txt"


def read_resume(path: Path) -> str:
    return docx2txt.process(str(path)) if path.exists() else ""


def read_summary(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def chunk_text(text: str, max_chars: int = 1500, overlap: int = 200) -> List[str]:
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def get_hf_embedder(hf_key: str, model_name: str) -> HuggingFaceEndpointEmbeddings:
    """Create an HF endpoint embedder using the modern langchain-huggingface wrapper."""
    return HuggingFaceEndpointEmbeddings(
        model=model_name,
        task="feature-extraction",
        huggingfacehub_api_token=hf_key,
    )


def build_faiss_index(docs: List[Dict[str, str]], embedder: HuggingFaceEndpointEmbeddings):
    texts: List[str] = []
    metadatas: List[Dict[str, str]] = []
    for doc in docs:
        texts.append(doc["text"])
        metadatas.append({"source": doc.get("source", "unknown"), "section": doc.get("section", "")})
    try:
        # Preferred: FAISS (fast ANN). Requires 'faiss-cpu' or 'faiss-gpu' installed.
        return FAISS.from_texts(texts=texts, embedding=embedder, metadatas=metadatas)
    except ImportError:
        # Fallback: pure-Python DocArray in-memory search (no native deps)
        if st is not None:
            st.warning(
                "FAISS not available (faiss-cpu/faiss-gpu not installed). Falling back to in-memory search.\n"
                "To enable FAISS, install a suitable faiss package for your OS."
            )
        return DocArrayInMemorySearch.from_texts(texts=texts, embedding=embedder, metadatas=metadatas)


PERSONA_SYSTEM = (
    "You are Gouranga Jha. Speak in first person as Gouranga (I/my). "
    "Answer strictly based on the provided context (resume and projects summary). "
    "Do not add generic disclaimers. "
    f"Only when the user's question requires information that is not present in the provided context, then add a single closing sentence: \"I don't have that in my notes right now—please check my portfolio at {PORTFOLIO_URL}.\" "
    "Never include this portfolio line for greetings or when the question is fully answered. "
    "Be concise, professional, and highlight relevant skills/tech where useful."
)

def validate_response(response_text: str, user_question: str) -> str:
    """
    Simple validation function to replace guardrails.
    Adds portfolio link only if the response seems incomplete or out of context.
    """
    # Keywords that suggest the question wasn't fully answered
    incomplete_indicators = [
        "i don't have information",
        "i cannot answer",
        "not available in my context",
        "i don't know",
        "not mentioned",
        "unable to provide"
    ]
    
    # Check if response suggests incomplete information
    response_lower = response_text.lower()
    is_incomplete = any(indicator in response_lower for indicator in incomplete_indicators)
    
    # Add portfolio link if response seems incomplete
    if is_incomplete and PORTFOLIO_URL not in response_text:
        return f"{response_text}\n\nFor more detailed information, please check my portfolio at {PORTFOLIO_URL}."
    
    return response_text


def run_llm(groq_key: str, prompt: str) -> Dict:
    try:
        # Try with a more standard model name first
        llm = ChatGroq(temperature=0.2, model_name="llama-3.1-8b-instant", groq_api_key=groq_key)
        # Use proper LangChain message format
        message = HumanMessage(content=prompt)
        raw = llm.invoke([message])
        return {"answer": raw.content, "sources": []}
    except Exception as e:
        # Fallback to original model name if the new one fails
        try:
            llm = ChatGroq(temperature=0.2, model_name="llama3-8b-8192", groq_api_key=groq_key)
            message = HumanMessage(content=prompt)
            raw = llm.invoke([message])
            return {"answer": raw.content, "sources": []}
        except Exception as fallback_error:
            # If both fail, return an error message
            return {"answer": f"Sorry, I'm having trouble processing your request right now. Error: {str(fallback_error)}", "sources": []}


def make_prompt(user_msg: str, context_chunks: List[str], history: List[Dict]) -> str:
    header = PERSONA_SYSTEM
    context = "\n\n".join([f"[Context {i+1}]\n{c}" for i, c in enumerate(context_chunks)])
    history_text = "\n".join(
        [f"{m['role'].capitalize()}: {m['content']}" for m in history[-6:]]
    )
    return (
        f"{header}\n\nContext:\n{context}\n\nConversation so far:\n{history_text}\n\n"
        f"User: {user_msg}\nAssistant:"
    )


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="💬", layout="wide")
    st.title(APP_TITLE)

    # Custom CSS to match portfolio gradient and light theme accents
    st.markdown(
        """
        <style>
        /* Page background gradient */
        .stApp {
            background: linear-gradient(180deg, #F5F2FF 0%, #EDE7FF 35%, #E8E6FF 70%, #FFFFFF 100%) !important;
        }
        /* Cards/containers subtle shadow */
        .stMarkdown, .stTextInput, .stChatInput, .stButton>button, .stExpander, .stSelectbox, .stTextArea {
            box-shadow: 0 2px 8px rgba(123, 97, 255, 0.08);
            border-radius: 10px;
        }
        /* Buttons primary */
        .stButton>button, .stChatInput>div>button {
            background: linear-gradient(90deg, #7B61FF 0%, #A78BFA 100%);
            color: #FFFFFF;
            border: none;
        }
        .stButton>button:hover, .stChatInput>div>button:hover {
            filter: brightness(0.95);
        }
        /* Sidebar */
        section[data-testid="stSidebar"] > div {
            background: #FFFFFF80;
            backdrop-filter: blur(6px);
        }
        /* Chat bubbles */
        .gj-bubble {
            padding: 14px 16px;
            border-radius: 12px;
            margin: 4px 0 8px 0;
            line-height: 1.55;
        }
        .gj-bubble.user {
            background: #FFFFFF;
            border: 1px solid #E0D7FF;
        }
        .gj-bubble.ai {
            background: #F3EEFF;
            border: 1px solid #D9CEFF;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    load_dotenv()
    with st.sidebar:
        st.subheader("Settings")
        if st.button("Clear chat"):
            st.session_state.pop("messages", None)
    top_k = 8

    resume_text = read_resume(RESUME_PATH)
    summary_text = read_summary(SUMMARY_PATH)

    docs: List[Dict[str, str]] = []
    for chunk in chunk_text(resume_text):
        docs.append({"text": chunk, "source": "Resume"})
    for chunk in chunk_text(summary_text):
        docs.append({"text": chunk, "source": "Projects_Summary"})

    # Only use HF_TOKEN as requested
    hf_key = os.getenv("HF_TOKEN", "")
    if not hf_key:
        st.error("Missing HF token. Set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN in your .env and restart.")
        return

    # Fixed free embedding model (no downloads via HF Inference API)
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedder = get_hf_embedder(hf_key, model_name)

    # Do NOT build the index on startup to avoid Streamlit Cloud health-check timeouts.
    # We will build it lazily on the first question.
    if "vs" not in st.session_state:
        st.session_state["vs"] = None

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Render history with custom bubble classes
    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            role_class = "user" if m["role"] == "user" else "ai"
            st.markdown(f"<div class='gj-bubble {role_class}'>" + m["content"] + "</div>", unsafe_allow_html=True)

    user_msg = st.chat_input("Ask me anything…")
    if user_msg:
        st.session_state["messages"].append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(f"<div class='gj-bubble user'>" + user_msg + "</div>", unsafe_allow_html=True)
        groq_key = os.getenv("GROQ_API_KEY", "")
        if not groq_key:
            st.error("Missing Groq key. Set GROQ_API_KEY in your .env and restart.")
            st.stop()

        # Build index on demand (first question only)
        if st.session_state.get("vs") is None:
            with st.spinner("Preparing knowledge index… (one-time per session)"):
                try:
                    st.session_state["vs"] = build_faiss_index(docs, embedder)
                except Exception as e:
                    st.error("Failed to prepare embeddings. Please verify HF_TOKEN and try again.\n\n" + str(e))
                    st.stop()

        # Retrieve via vector search
        docs_found = st.session_state["vs"].similarity_search(user_msg, k=top_k)
        retrieved = [d.page_content for d in docs_found]
        sources = list({d.metadata.get("source", "") for d in docs_found}) if docs_found else []

        prompt = make_prompt(user_msg, retrieved, st.session_state["messages"])
        raw_response = run_llm(groq_key, prompt)

        # Simple validation to add portfolio link when needed
        result_text = raw_response.get("answer", "")
        result_text = validate_response(result_text, user_msg)

        st.session_state["messages"].append({"role": "assistant", "content": result_text})
        with st.chat_message("assistant"):
            st.markdown(f"<div class='gj-bubble ai'>" + result_text + "</div>", unsafe_allow_html=True)
        with st.expander("Sources"):
            if sources:
                st.write(", ".join(sources))
            else:
                st.write("(Resume / Projects Summary)")


if __name__ == "__main__":
    main()


