# Professional AI Research Assistant

A complete local Streamlit research workspace with a professional glassmorphism interface.

## AI backend

- Sentence Transformers: semantic embeddings
- FAISS: vector retrieval
- FLAN-T5: grounded QA and academic generation
- NLLB-200: multilingual translation
- Pandas: dataset question answering and analysis
- Plotly: interactive visualizations
- SQLite: document and note metadata
- gTTS: voice output

## Features

- Document upload and indexing
- Selected-document semantic search
- TF-IDF search
- Document question answering
- Dataset question answering
- Map-reduce research-paper summarization
- Academic abstract generation
- TF-IDF keyword/key-phrase extraction
- Exact-count research-question generation
- Translation
- Citation generation
- Literature review
- Paper comparison and similarity
- Dataset analytics
- Notes
- Dashboard
- TXT, DOCX and PDF export

## Run on Windows PowerShell

```powershell
cd "<project folder>"
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The first use of Sentence Transformers, FLAN-T5 or NLLB-200 downloads model files.
Python 3.11 or 3.12 is recommended for the broadest library compatibility.
