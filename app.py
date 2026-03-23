import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from pdf_parser import process_pdf
from embeddings import compare_chunks
from llm_analyzer import categorize_changes, generate_summary

st.set_page_config(layout="wide", page_title="Semantic PDF Comparison")

st.title("Semantic PDF Document Comparison")
st.markdown("Upload two versions of a PDF to see what meaningfully changed.")

col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("Upload Version A (Original)", type=["pdf"])
with col2:
    file_b = st.file_uploader("Upload Version B (Modified)", type=["pdf"])

if st.button("Compare Documents"):
    if file_a is None or file_b is None:
        st.warning("Please upload both PDF files.")
    elif not os.environ.get("GROQ_API_KEY"):
        st.error("GROQ_API_KEY is not set in the environment or `.env` file.")
    else:
        with st.spinner("Extracting and parsing PDFs..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_a:
                tmp_a.write(file_a.read())
                path_a = tmp_a.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_b:
                tmp_b.write(file_b.read())
                path_b = tmp_b.name
            
            chunks_a = process_pdf(path_a)
            chunks_b = process_pdf(path_b)
            
            os.remove(path_a)
            os.remove(path_b)
            
        with st.spinner("Computing semantic similarities..."):
            base_comparisons = compare_chunks(chunks_a, chunks_b)
            
        with st.spinner("Analyzing changes with Groq LLM..."):
            final_comparisons = categorize_changes(base_comparisons)
            summary = generate_summary(final_comparisons)
            
        st.success("Comparison complete!")
        
        st.header("Executive Summary")
        st.write(summary)
        st.divider()
        
        st.header("Detailed Side-by-Side Comparison")
        
        # Color coding map
        color_map = {
            "Unchanged": "#d4edda",        # Green
            "Formatting Only": "#d4edda",  # Green
            "Added": "#cce5ff",            # Blue
            "Removed": "#f8d7da",          # Red
            "Modified": "#fff3cd",         # Yellow
            "Modified (LLM Error)": "#e2e3e5" # Grayish
        }
        
        for idx, item in enumerate(final_comparisons):
            status = item["status"]
            sim = item.get("similarity", 0.0)
            color = color_map.get(status, "#ffffff")
            
            st.markdown(f"**Section {idx+1} | Status: {status} | Similarity: {sim:.2f}**")
            
            cc1, cc2 = st.columns(2)
            with cc1:
                content_a = item['chunk_a'] or "<i>[No Content]</i>"
                st.markdown(f"<div style='background-color: {color}; color: #333; padding: 15px; border-radius: 5px; height: 100%; border: 1px solid #ccc; font-family: sans-serif; white-space: pre-wrap;'>{content_a}</div>", unsafe_allow_html=True)
            with cc2:
                content_b = item['chunk_b'] or "<i>[No Content]</i>"
                st.markdown(f"<div style='background-color: {color}; color: #333; padding: 15px; border-radius: 5px; height: 100%; border: 1px solid #ccc; font-family: sans-serif; white-space: pre-wrap;'>{content_b}</div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
