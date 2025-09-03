# scripts/generate_qa_dataset_gemini.py
import google.generativeai as genai
import json
import os
import time
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
load_dotenv() # Load environment variables from .env file
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load your document chunks
try:
    with open('data/processed/document_chunks.json', 'r') as f:
        document_chunks = json.load(f)
except FileNotFoundError:
    print("Error: document_chunks.json not found. Please run the previous processing step.")
    exit()

def generate_qa_pair(chunk_data):
    """Generates a question-answer pair from a single document chunk using Gemini."""
    content = chunk_data.get("content", "")
    source = chunk_data.get("source", "Unknown")

    if len(content) < 150: # Skip very short chunks that lack context
        return None

    # Consolidated prompt for Gemini
    prompt = f"""
    You are an expert in EPFO rules and regulations. Your task is to generate a highly relevant question-and-answer pair based *only* on the provided text context.
    The question should be a realistic user query that can be answered using the context. The answer must be factually accurate and derived directly from the text.
    Do not add any information not present in the context.

    Context:
    ---
    {content}
    ---

    Output the result in a strict JSON format with keys "instruction" and "response".
    Example Output:
    {{
        "instruction": "What is the deadline for filing the annual return?",
        "response": "According to the provided text, the deadline for filing the annual return is March 31st."
    }}
    """

    try:
        # Configure the model and generation settings
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.5,
                response_mime_type="application/json"
            )
        )
        
        # The response text should be a valid JSON string
        qa_pair = json.loads(response.text)
        
        # Add source for traceability
        qa_pair['source'] = source
        return qa_pair

    except Exception as e:
        print(f"Error generating QA for chunk from {source}: {e}")
        # This can happen due to API errors, content filtering, or invalid JSON output
        return None

# --- Main Execution ---
qa_dataset = []
# Use ThreadPoolExecutor for faster processing
with ThreadPoolExecutor(max_workers=5) as executor: # Note: A lower worker count is safer for Gemini API rate limits
    futures = [executor.submit(generate_qa_pair, chunk) for chunk in document_chunks]
    for i, future in enumerate(as_completed(futures)):
        result = future.result()
        if result:
            qa_dataset.append(result)
            print(f"Generated QA pair #{len(qa_dataset)} from chunk {i+1}/{len(document_chunks)}")
        
        # A small delay to respect API rate limits
        time.sleep(1) 

# Save the generated dataset
output_path = 'data/training/epfo_qa_dataset.json'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(qa_dataset, f, indent=4)

print(f"\nSuccessfully generated and saved {len(qa_dataset)} Q&A pairs.")