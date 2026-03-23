import fitz  # PyMuPDF
import pdfplumber
import re
from typing import List

def extract_text_and_tables(pdf_path: str) -> str:
    """Extracts text and table data from a PDF file."""
    text_content = []
    
    # Extract text using PyMuPDF
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text = page.get_text("text")
            if text:
                text_content.append(text)
        doc.close()
    except Exception as e:
        print(f"Error reading with fitz for {pdf_path}: {e}")
        
    # Extract tables using pdfplumber
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        # Convert table to a readable string format
                        for row in table:
                            row_text = " | ".join([str(cell).strip() if cell else "" for cell in row])
                            text_content.append(row_text)
                        text_content.append("\n") # Add spacing after table
    except Exception as e:
        print(f"Error reading with pdfplumber for {pdf_path}: {e}")
        
    return "\n".join(text_content)

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Chunks text into meaningful paragraphs or sections."""
    # Split by double newlines to try and maintain paragraph boundaries
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        para_len = len(para)
        if current_length + para_len > chunk_size and current_chunk:
            chunks.append("\n".join(current_chunk))
            # Start new chunk with overlap if needed
            current_chunk = [current_chunk[-1]] if overlap > 0 and len(current_chunk) > 0 else []
            current_length = sum(len(c) for c in current_chunk)
            
        current_chunk.append(para)
        current_length += para_len
        
    if current_chunk:
        chunks.append("\n".join(current_chunk))
        
    return chunks

def process_pdf(pdf_path: str) -> List[str]:
    """Full pipeline to extract and chunk a PDF."""
    full_text = extract_text_and_tables(pdf_path)
    chunks = chunk_text(full_text)
    return chunks
