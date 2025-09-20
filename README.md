Gouranga Jha – RAG Chatbot (Portfolio Assistant)

A Streamlit app that answers in first person as Gouranga Jha using Retrieval‑Augmented Generation (RAG) over:

- data/Gouranga_Jha_Resume.docx
- data/Projects_Summary.txt

Embeddings are computed via Hugging Face Inference API (no local downloads) and indexed in‑memory with FAISS per session. The UI is theme‑matched to the portfolio and runs well on Streamlit Cloud.

Features
- First‑person persona (Gouranga Jha) with concise, grounded answers
- RAG over resume + project summary
- Hugging Face endpoint embeddings (sentence-transformers/all-MiniLM-L6-v2)
- FAISS vector search (auto‑fallback to in‑memory search if FAISS not available)
- Simple validation: add portfolio link only when the answer is out of context
- Streamlit chat UI with history; Top‑K is fixed to 8
- Light theme + gradient styling aligned with portfolio

Quickstart (local)
1) Create & activate venv (Windows)
   - python -m venv venv
   - venv\Scripts\activate
2) Install deps
   - pip install -r requirements.txt
3) Add environment variables (create .env)
   - GROQ_API_KEY=...
   - HF_TOKEN=...
4) Run
   - streamlit run app.py

Environment variables
- GROQ_API_KEY=your_groq_key
- HF_TOKEN=your_huggingface_api_token

Notes
- HF token needs read scope; the app uses sentence-transformers/all-MiniLM-L6-v2 via the HF Inference API.
- No keys are shown in the UI.

How it works
1) Loads and chunks data/Gouranga_Jha_Resume.docx and data/Projects_Summary.txt (≈1500 chars, 200 overlap).
2) Generates embeddings through HF Inference API and builds an in‑memory FAISS index for the session.
3) For each chat turn, retrieves Top‑K (8) chunks and sends the context + short conversation history to the LLM (Groq).
4) Simple validation post‑processes: if the answer requires info outside the context, a single closing sentence invites the user to check the portfolio.

Deploy to Streamlit Cloud
1) Push this repo to GitHub.
2) On Streamlit Cloud, create a new app:
   - Repository: Gouranga-GH/my_portfolio_cb
   - Branch: main
   - Main file: app.py
3) In App → Settings → Secrets, add:
   - GROQ_API_KEY = ...
   - HF_TOKEN = ...
4) Deploy. The app builds the index each session (no persistent storage required).

Project structure
- app.py – entrypoint (launches the Streamlit app)
- apps/gj_rag_chat/main.py – Streamlit app logic (RAG, UI, validation)
- .streamlit/config.toml – Light theme that matches portfolio
- requirements.txt – Python deps
- sample_env.txt – Example env values
- data – Resume + Projects_Summary

License
- MIT

