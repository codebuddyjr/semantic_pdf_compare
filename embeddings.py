from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from typing import List, Dict

# Standardize on all-MiniLM-L6-v2 as requested
MODEL_NAME = 'all-MiniLM-L6-v2'
model = None

def get_model():
    """Lazy evaluation to avoid loading the model until needed."""
    global model
    if model is None:
        model = SentenceTransformer(MODEL_NAME)
    return model

def compute_embeddings(chunks: List[str]) -> torch.Tensor:
    """Computes embeddings for a list of text chunks."""
    if not chunks:
        return torch.tensor([])
    m = get_model()
    embeddings = m.encode(chunks, convert_to_tensor=True)
    return embeddings

def compare_chunks(chunks_a: List[str], chunks_b: List[str], threshold: float = 0.85) -> List[Dict]:
    """
    Compares two lists of chunks and finds matching/changed sections.
    Uses a simple greedy sequence alignment matching text segments.
    """
    if not chunks_a and not chunks_b:
        return []
        
    if not chunks_a:
        return [{"chunk_a": None, "chunk_b": b, "similarity": 0.0, "status": "Semantically Changed"} for b in chunks_b]
    if not chunks_b:
        return [{"chunk_a": a, "chunk_b": None, "similarity": 0.0, "status": "Semantically Changed"} for a in chunks_a]

    embs_a = compute_embeddings(chunks_a)
    embs_b = compute_embeddings(chunks_b)
    
    cosine_scores = util.cos_sim(embs_a, embs_b)
    
    results = []
    i, j = 0, 0
    
    while i < len(chunks_a) and j < len(chunks_b):
        score = cosine_scores[i][j].item()
        
        if score >= threshold:
            results.append({
                "chunk_a": chunks_a[i], 
                "chunk_b": chunks_b[j], 
                "similarity": score, 
                "status": "Unchanged"
            })
            i += 1
            j += 1
        else:
            # Look ahead for insertions and deletions
            match_found = False
            for offset in range(1, 4):
                # Check for insertions in B
                if j + offset < len(chunks_b) and cosine_scores[i][j+offset].item() >= threshold:
                    for k in range(offset):
                        results.append({
                            "chunk_a": None, 
                            "chunk_b": chunks_b[j+k], 
                            "similarity": 0.0, 
                            "status": "Semantically Changed"
                        })
                    j += offset
                    match_found = True
                    break
                # Check for deletions in A
                if i + offset < len(chunks_a) and cosine_scores[i+offset][j].item() >= threshold:
                    for k in range(offset):
                        results.append({
                            "chunk_a": chunks_a[i+k], 
                            "chunk_b": None, 
                            "similarity": 0.0, 
                            "status": "Semantically Changed"
                        })
                    i += offset
                    match_found = True
                    break
            
            if not match_found:
                # Treat as a modification/replacement of current block
                results.append({
                    "chunk_a": chunks_a[i], 
                    "chunk_b": chunks_b[j], 
                    "similarity": score, 
                    "status": "Semantically Changed"
                })
                i += 1
                j += 1

    # Add any remaining chunks
    while i < len(chunks_a):
        results.append({
            "chunk_a": chunks_a[i], 
            "chunk_b": None, 
            "similarity": 0.0, 
            "status": "Semantically Changed"
        })
        i += 1
        
    while j < len(chunks_b):
        results.append({
            "chunk_a": None, 
            "chunk_b": chunks_b[j], 
            "similarity": 0.0, 
            "status": "Semantically Changed"
        })
        j += 1
        
    return results
