# Intelligent Document Assistant

A retrieval-augmented generation (RAG) chatbot for asking questions across PDF,
TXT, and DOCX files. Documents are parsed and chunked locally, embedded with a
Hugging Face Sentence Transformers model, and searched in an in-memory FAISS
index. Google Gemini answers with retrieved passages as its only evidence.

## Features

- Upload multiple PDF, TXT, and DOCX files (up to 20 MB per file).
- Preserve PDF page numbers and source filenames in chunk metadata.
- Split text with LangChain's `RecursiveCharacterTextSplitter` (1,000-character
  chunks with 200-character overlap).
- Create local embeddings with `sentence-transformers/all-MiniLM-L6-v2`; no paid
  embedding API is needed.
- Retrieve top-k relevant passages from FAISS and show source passages below
  each answer.
- Reject low-relevance results using a configurable maximum L2 distance.
- Keep chat history, reset the conversation, and export it as TXT or JSON.
- Rebuild or clear the current document collection.
- Keep uploaded files and indexes in memory for the active Streamlit session.
  The application does not deserialize indexes or write uploaded documents to
  disk.

## Architecture

1. `src/document_loader.py` extracts text from each format and records page
   numbers where provided by PDF.
2. `src/text_splitter.py` creates overlapping LangChain document chunks and
   assigns chunk identifiers.
3. `src/embeddings.py` loads and caches the configured Sentence Transformers
   embedding model locally.
4. `src/vector_store.py` builds an in-memory FAISS index and filters similarity
   results by L2 distance.
5. `src/rag_pipeline.py` supplies only retrieved passages and the current
   question to Gemini. Document passages are explicitly treated as untrusted
   data, not instructions. Conversation history is never used as evidence.
6. `app.py` provides document management, configuration, chat, source inspection,
   and downloads through Streamlit.

## Technology

- Python 3.11
- Streamlit
- LangChain document types and recursive text splitting
- FAISS
- Hugging Face Sentence Transformers
- Google Gemini API (`google-genai`)
- PyPDF and python-docx

## Installation

Use Python 3.11. Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The Sentence Transformers model is downloaded from Hugging Face on first use and
then cached locally. An internet connection is needed for that first download.
FAISS wheels vary by operating system; if `faiss-cpu` has no compatible wheel
for your platform, install a supported FAISS build before launching the app.

## Gemini API setup

Copy `.env.example` to `.env` and add a Gemini API key:

```env
GOOGLE_API_KEY=your_google_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
```

The API key is read from the environment and must not be committed. Change
`GEMINI_MODEL` to another model available to your API account if needed. The
application reports API errors rather than silently switching models.

## Run the application

```powershell
streamlit run app.py
```

Upload documents from the sidebar and select **Process documents**. After
processing completes, ask questions in the chat input. Use the sidebar sliders
to adjust the result count and maximum FAISS L2 distance. The embedding model
field can be changed to another compatible Hugging Face Sentence Transformers
model; changing it rebuilds the index.

## Deployment

For Streamlit Community Cloud, push the project to a Git repository, select
`app.py` as the application entry point, and configure `GOOGLE_API_KEY` (and
optionally `GEMINI_MODEL`) in the app's Secrets settings. The service installs
the dependencies from `requirements.txt`. Keep in mind that uploads and the
FAISS index are held in memory and are not shared across sessions or preserved
after a session restarts.

## Document processing and search

Text extraction and embedding happen locally. PDFs are split into page-level
documents so page numbers remain available; TXT and DOCX files retain their
filename. Empty or unreadable files show a processing error. All extracted
documents are chunked with a 1,000-character size and 200-character overlap.

FAISS uses normalized Sentence Transformers embeddings and Euclidean (L2)
distance. Lower distances are closer matches. The initial maximum distance is
1.5; tune it for the embedding model and your documents. If no retrieved chunks
meet the selected threshold, the app returns a clear insufficient-context
response without calling Gemini.

Each answer includes an expandable list of retrieved passages and their
filename/page references. Retrieved sources are fresh for each question;
previous chat turns are not included as evidence.

## Tests

Run the offline unit tests with:

```powershell
python -m pytest -q
```

The tests mock external Gemini calls and use small fixtures. Real embedding
model download and live Gemini API access are not required for the suite.

## Screenshots

Screenshots can be added here after running the application locally.

## Limitations and future improvements

- The FAISS index and uploaded documents are session-scoped and in-memory; they
  are cleared when the Streamlit session ends or the user clears documents.
- Scanned PDFs require OCR, which is not included.
- L2 distance thresholds are model- and document-dependent and may need tuning.
- A production deployment should add authentication, storage isolation,
  resource quotas, monitoring, and an OCR pipeline where required.
