# Semantic PDF Document Comparison Tool

A Streamlit web application that compares two PDF documents, semantically chunks their text, and uses the Groq API (`llama-3.3-70b-versatile`) to classify and summarize the differences.

## Features
- Upload two PDFs (Version A and Version B).
- Text and table extraction using `PyMuPDF` and `pdfplumber`.
- Semantic similarity comparison using `sentence-transformers` (`all-MiniLM-L6-v2`).
- LLM summarization and intelligent change categorization (Added, Removed, Modified, Formatting Only) via Groq API.
- Modern color-coded Streamlit UI.

## Setup Instructions

1. **Clone the repository or navigate to this folder.**
2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up your environment variables:**
   - Copy `.env.example` to `.env`.
   - Add your `GROQ_API_KEY` to the `.env` file.
5. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## Usage
- Open the application in your browser (usually `http://localhost:8501`).
- Upload two PDFs and click **Compare Documents**.
- Review the executive summary at the top and step through the side-by-side chunks below.

## Interface Example
*(Insert screenshot here)*


## LIMITATION
- The current sequential alignment approach struggles when content is reordered between versions rather than modified in place.
