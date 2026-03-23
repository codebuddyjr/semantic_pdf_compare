import os
from groq import Groq
from typing import List, Dict

# Initialize Groq client
client = None

def get_client():
    global client
    if client is None:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return client

def categorize_changes(changes: List[Dict]) -> List[Dict]:
    """
    Takes a list of changed chunks and categorizes them using Groq LLM.
    """
    if not changes:
        return []
        
    groq_client = get_client()
    
    # We only process items flagged as "Semantically Changed"
    for item in changes:
        if item["status"] != "Semantically Changed":
            continue
            
        a_text = item["chunk_a"] or "[NO TEXT]"
        b_text = item["chunk_b"] or "[NO TEXT]"
        
        prompt = f"""You are a precise document comparison assistant. 
Review the following paired text sections comparing Version A to Version B.
Version A:
{a_text}
Version B:
{b_text}

Categorize the change that occurred from A to B into exactly ONE of the following:
- Added (if A is mostly empty/missing and B has new content)
- Removed (if A has content and B is mostly empty/missing)
- Modified (if content changed meaning)
- Formatting Only (if the meaning is exactly the same but wording/spacing/formatting differs)

Return ONLY the classification string and nothing else."""

        try:
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.0,
                max_tokens=20,
            )
            classification = chat_completion.choices[0].message.content.strip()
            
            # Map robustly
            if "Add" in classification:
                item["status"] = "Added"
            elif "Remov" in classification:
                item["status"] = "Removed"
            elif "Formatt" in classification:
                item["status"] = "Formatting Only"
            else:
                item["status"] = "Modified"
                
        except Exception as e:
            print(f"Error calling Groq: {e}")
            item["status"] = "Modified (LLM Error)"
            
    return changes

def generate_summary(changes: List[Dict]) -> str:
    """Generates an executive summary of the document differences."""
    meaningful_changes = [c for c in changes if c["status"] in ["Added", "Removed", "Modified"]]
    
    if not meaningful_changes:
        return "No meaningful semantic changes were found between the documents."
        
    summary_input = ""
    for i, c in enumerate(meaningful_changes[:20]): # Limit to first 20 to fit prompt easily
        summary_input += f"Change {i+1} ({c['status']}):\nVersion A: {c['chunk_a']}\nVersion B: {c['chunk_b']}\n\n"
        
    prompt = f"""You are a precise document comparison assistant.
Below are the most significant changes identified between two versions of a document.
Please provide a high-level executive summary (3-4 paragraphs maximum) describing what meaningfully changed overall.
Focus on the business or semantic impact of replacements, additions, and deletions.

Changes:
{summary_input}
"""
    
    try:
        groq_client = get_client()
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=600,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Warning: Could not generate executive summary due to LLM error: {e}"
