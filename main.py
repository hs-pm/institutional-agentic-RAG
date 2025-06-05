import os
import json
import faiss
import uvicorn
import hashlib
import requests
import threading
import time
import numpy as np  # Fixed: Move numpy import to top
import logging
from datetime import datetime
from typing import List
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from threading import Lock

# --- CONFIG ---
DATA_DIR = "./data"
GROQ_API_KEY = "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"
GROQ_MODEL = "llama3-70b-8192"
INDEX_FILE = "faiss.index"
LOG_FILE = "rag_api_flow.log"
model = SentenceTransformer('all-MiniLM-L6-v2')
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create router and include it
router = APIRouter()
app.include_router(router)

@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration"""
    try:
        # Check if index is initialized
        if not index:
            return {"status": "degraded", "message": "Search index not initialized"}
        
        # Check data directory
        if not os.path.exists(DATA_DIR):
            return {"status": "degraded", "message": "Data directory not accessible"}
            
        return {
            "status": "healthy",
            "index_size": index.ntotal if index else 0,
            "chunks": len(all_text_chunks),
            "documents": len(txt_docs),
            "metadata": len(doc_metadata)
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.on_event("startup")
async def startup_event():
    """Initialize data and index when the server starts"""
    load_all_data()
    start_file_watcher()

# --- LOGGING SETUP ---
def setup_logging():
    """Setup dual logging to console and file"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, mode='w'),  # Overwrite log file each run
            logging.StreamHandler()  # Console output
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def log_step(step_name, details=None, status="SUCCESS"):
    """Log each step with timestamp and details"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    message = f"STEP: {step_name} | STATUS: {status}"
    if details:
        message += f" | DETAILS: {details}"
    logger.info(message)
    print(f"[{timestamp}] {message}")  # Also print to console for immediate feedback

# --- STATE ---
lock = Lock()
txt_docs, doc_metadata, slack_data, all_text_chunks, chunk_hashes = [], [], [], [], set()
index = None

# --- UTILS ---
def chunk_text(text, chunk_size=512, overlap=50):
    log_step("TEXT_CHUNKING_START", f"text_length={len(text)}, chunk_size={chunk_size}, overlap={overlap}")
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    log_step("TEXT_CHUNKING_COMPLETE", f"created {len(chunks)} chunks")
    return chunks
def filter_chunks(all_chunks, metadata, author=None, platform=None, start_date=None, end_date=None):
    # Use same filtering logic as metadata
    filtered_items = filter_metadata(metadata, author, platform, start_date, end_date)

    # Now, select chunks that match those filtered items
    titles = {item['title'] for item in filtered_items if 'title' in item}
    return [chunk for chunk in all_chunks if any(title in chunk for title in titles)], filtered_items


def hash_text(text):
    log_step("HASH_TEXT_START", f"text_length={len(text)}")
    hash_value = hashlib.md5(text.encode('utf-8')).hexdigest()
    log_step("HASH_TEXT_COMPLETE", f"hash={hash_value[:8]}...")
    return hash_value

# --- SCHEMA ---
class SearchInput(BaseModel):
    query: str = ""
    context: str = ""
    author: str = None
    platform: str = None  # e.g., "slack", "document"
    start_date: str = None  # ISO date: "2024-01-01"
    end_date: str = None

from datetime import datetime

def filter_metadata(data, author=None, platform=None, start_date=None, end_date=None):
    def matches(item):
        if author and author.lower() not in item.get("author", "").lower():
            return False
        if platform and platform.lower() not in item.get("platform", "").lower():
            return False
        date_str = item.get("date", "")  # expected ISO format
        if start_date and date_str:
            try:
                item_date = datetime.fromisoformat(date_str)
                if datetime.fromisoformat(start_date) > item_date:
                    return False
            except ValueError:
                return False
        if end_date and date_str:
            try:
                item_date = datetime.fromisoformat(date_str)
                if datetime.fromisoformat(end_date) < item_date:
                    return False
            except ValueError:
                return False
        return True

    return [item for item in data if matches(item)]


# --- SUMMARIZATION ---
def generate_summary(query, context, matched_text):
    log_step("LLM_SUMMARY_START", f"query_length={len(query)}, context_length={len(context)}, matched_text_length={len(matched_text)}")
    
    prompt = f"""You are a summarizer. Based on the following query and context:

Query: {query}
Context: {context}

Matched data:
{matched_text}

You are an expert summarizer working for a crypto trading enterprise. Your goal is to analyze internal discussions, documents, and performance reports to generate concise executive-level summaries for institutional memory and strategy planning.
"""
    
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {
        "model": GROQ_MODEL,
        "max_tokens": 1024,
        "messages": [
            {"role": "system", "content": "You summarize trading and performance data."},
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        log_step("LLM_API_CALL_START", f"model={GROQ_MODEL}, max_tokens=1024")
        res = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)
        log_step("LLM_API_CALL_COMPLETE", f"status_code={res.status_code}")
        
        summary = res.json()["choices"][0]["message"]["content"]
        log_step("LLM_SUMMARY_COMPLETE", f"summary_length={len(summary)}")
        return summary
    except Exception as e:
        log_step("LLM_SUMMARY_ERROR", f"error={str(e)}", "ERROR")
        return f"[ERROR] LLM call failed: {e}"

# --- LOADERS ---
def parse_txt_file(filepath):
    log_step("PARSE_TXT_FILE_START", f"filepath={filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        result = {"title": os.path.basename(filepath), "full_text": content}
        log_step("PARSE_TXT_FILE_COMPLETE", f"title={result['title']}, content_length={len(content)}")
        return result
    except Exception as e:
        log_step("PARSE_TXT_FILE_ERROR", f"filepath={filepath}, error={str(e)}", "ERROR")
        print(f"[ERROR] Reading {filepath}: {e}")
        return None

def load_json(file, key):
    log_step("LOAD_JSON_START", f"file={file}, key={key}")
    try:
        filepath = os.path.join(DATA_DIR, file)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        result = data.get(key, [])
        log_step("LOAD_JSON_COMPLETE", f"file={file}, loaded_items={len(result)}")
        return result
    except Exception as e:
        log_step("LOAD_JSON_ERROR", f"file={file}, error={str(e)}", "ERROR")
        print(f"[ERROR] Loading {file}: {e}")
        return []

def load_all_data():
    global txt_docs, doc_metadata, slack_data, index, all_text_chunks, chunk_hashes
    
    log_step("LOAD_ALL_DATA_START", "Starting data loading process")
    
    try:
        with lock:
            log_step("LOCK_ACQUIRED", "Data loading lock acquired")
            
            # Initialize data structures
            txt_docs, all_text_chunks, chunk_hashes = [], [], set()
            new_vectors = []
            log_step("DATA_STRUCTURES_INIT", "Initialized empty data structures")
            
            # Load JSON metadata
            log_step("JSON_METADATA_LOAD_START", "Loading JSON metadata files")
            doc_metadata = load_json("documents.json", "documents")
            slack_data = load_json("slack_discussions.json", "discussions")
            log_step("JSON_METADATA_LOAD_COMPLETE", f"doc_metadata={len(doc_metadata)}, slack_data={len(slack_data)}")

            # Load text files
            log_step("TXT_FILES_LOAD_START", f"Scanning directory: {DATA_DIR}")
            txt_file_count = 0
            if os.path.exists(DATA_DIR):
                for fname in os.listdir(DATA_DIR):
                    if fname.endswith(".txt"):
                        log_step("TXT_FILE_PROCESSING", f"processing file: {fname}")
                        doc = parse_txt_file(os.path.join(DATA_DIR, fname))
                        if doc:
                            txt_docs.append(doc)
                            txt_file_count += 1
                            
                            log_step("TXT_FILE_CHUNKING_START", f"chunking file: {fname}")
                            chunks = chunk_text(doc['full_text'])
                            for chunk_idx, chunk in enumerate(chunks):
                                text = doc['title'] + "\n" + chunk
                                text_hash = hash_text(text)
                                if text_hash not in chunk_hashes:
                                    all_text_chunks.append(text)
                                    chunk_hashes.add(text_hash)
                                    vector = model.encode(text)
                                    new_vectors.append(vector)
                                    log_step("CHUNK_ENCODED", f"file={fname}, chunk={chunk_idx}, vector_dim={len(vector)}")
            
            log_step("TXT_FILES_LOAD_COMPLETE", f"processed {txt_file_count} text files, total_chunks={len(all_text_chunks)}")

            # Process metadata for vectorization
            log_step("METADATA_VECTORIZATION_START", "Processing metadata for vectorization")
            metadata_vectors = 0
            for item_idx, item in enumerate(doc_metadata + slack_data):
                combined = f"{item.get('title', '')} {item.get('description', '')}"
                text_hash = hash_text(combined)
                if text_hash not in chunk_hashes:
                    all_text_chunks.append(combined)
                    chunk_hashes.add(text_hash)
                    vector = model.encode(combined)
                    new_vectors.append(vector)
                    metadata_vectors += 1
                    log_step("METADATA_ENCODED", f"item={item_idx}, title={item.get('title', 'N/A')[:30]}...")
            
            log_step("METADATA_VECTORIZATION_COMPLETE", f"processed {metadata_vectors} metadata items")

            # FAISS index handling
            log_step("FAISS_INDEX_SETUP_START", f"Setting up FAISS index with {len(new_vectors)} vectors")
            
            if new_vectors:
                new_vectors = np.array(new_vectors).astype('float32')
                dim = new_vectors.shape[1]
                log_step("VECTORS_PREPARED", f"vectors_shape={new_vectors.shape}, dimension={dim}")
                
                # Load existing index or create new one
                if os.path.exists(INDEX_FILE):
                    try:
                        log_step("EXISTING_INDEX_LOAD_START", f"Loading existing index: {INDEX_FILE}")
                        index = faiss.read_index(INDEX_FILE)
                        log_step("EXISTING_INDEX_LOAD_COMPLETE", f"loaded index with {index.ntotal} existing vectors")
                    except Exception as e:
                        log_step("EXISTING_INDEX_LOAD_ERROR", f"error={str(e)}", "ERROR")
                        index = None
                
                # Create new index if needed
                if index is None:
                    num_vectors = len(new_vectors)
                    log_step("NEW_INDEX_CREATE_START", f"Creating new index for {num_vectors} vectors")
                    
                    if num_vectors < 100:
                        # Use simple flat index for small datasets
                        index = faiss.IndexFlatL2(dim)
                        log_step("FLAT_INDEX_CREATED", f"created flat index, dimension={dim}")
                    else:
                        # Use IVF index for larger datasets
                        nlist = min(100, num_vectors // 10)  # Adaptive number of clusters
                        quantizer = faiss.IndexFlatL2(dim)
                        index = faiss.IndexIVFFlat(quantizer, dim, nlist)
                        log_step("IVF_INDEX_TRAINING_START", f"training IVF index with {nlist} clusters")
                        index.train(new_vectors)
                        log_step("IVF_INDEX_TRAINING_COMPLETE", f"IVF index trained successfully")
                
                # Add vectors to index
                log_step("INDEX_ADD_VECTORS_START", f"adding {len(new_vectors)} vectors to index")
                index.add(new_vectors)
                log_step("INDEX_ADD_VECTORS_COMPLETE", f"index now contains {index.ntotal} total vectors")
                
                # Save index
                try:
                    log_step("INDEX_SAVE_START", f"saving index to {INDEX_FILE}")
                    faiss.write_index(index, INDEX_FILE)
                    log_step("INDEX_SAVE_COMPLETE", f"index saved successfully")
                except Exception as e:
                    log_step("INDEX_SAVE_ERROR", f"error={str(e)}", "ERROR")
            else:
                # No vectors to index
                log_step("NO_VECTORS_FALLBACK", "No vectors found, creating empty flat index")
                index = faiss.IndexFlatL2(384)  # Default dimension for all-MiniLM-L6-v2

            log_step("LOAD_ALL_DATA_COMPLETE", f"txt_docs={len(txt_docs)}, doc_metadata={len(doc_metadata)}, slack_data={len(slack_data)}, total_chunks={len(all_text_chunks)}")
            
    except Exception as e:
        log_step("LOAD_ALL_DATA_ERROR", f"critical_error={str(e)}", "ERROR")
        print(f"[ERROR] Failed to load data: {e}")
        # Create minimal fallback index
        index = faiss.IndexFlatL2(384)
        all_text_chunks = []
        log_step("FALLBACK_INDEX_CREATED", "created fallback empty index")

# --- WATCHER ---
class ChangeHandler(FileSystemEventHandler):
    def on_any_event(self, event):
        if event.src_path.endswith(".txt") or event.src_path.endswith(".json"):
            log_step("FILE_CHANGE_DETECTED", f"file={event.src_path}, event_type={event.event_type}")
            print(f"[INFO] File changed: {event.src_path}")
            load_all_data()

def start_file_watcher():
    log_step("FILE_WATCHER_START", f"watching directory: {DATA_DIR}")
    try:
        observer = Observer()
        observer.schedule(ChangeHandler(), path=DATA_DIR, recursive=False)
        observer.start()
        log_step("FILE_WATCHER_ACTIVE", "file watcher started successfully")
        print("[INFO] File watcher active.")
    except Exception as e:
        log_step("FILE_WATCHER_ERROR", f"error={str(e)}", "ERROR")
        print(f"[WARNING] Failed to start file watcher: {e}")
from collections import Counter
import re

def extract_top_keywords(text, top_n=5):
    words = re.findall(r'\b\w+\b', text.lower())
    stopwords = {'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'a', 'an', 'this', 'that'}
    words = [w for w in words if len(w) > 2 and w not in stopwords]
    freq = Counter(words)
    return [word for word, count in freq.most_common(top_n)]

# --- API ---
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime

@app.post("/search")
def search(input: SearchInput):
    log_step("API_SEARCH_REQUEST", f"query='{input.query}', context_length={len(input.context)}")

    try:
        with lock:
            log_step("API_LOCK_ACQUIRED", "search lock acquired")

            if not index:
                log_step("API_VALIDATION_ERROR", "index not initialized", "ERROR")
                return {"error": "Index not initialized."}

            query = input.query.strip()
            context = input.context.strip()
            log_step("API_INPUT_PROCESSED", f"cleaned_query='{query}', context_length={len(context)}")

            if not query:
                log_step("API_VALIDATION_ERROR", "empty query provided", "ERROR")
                return {"error": "Query cannot be empty."}

            if not all_text_chunks:
                log_step("API_EMPTY_INDEX", "no indexed content available")
                return {
                    "summary": {
                        "text": "No indexed content available.",
                        "matchCount": {"total": 0, "slack": 0, "docs": 0},
                        "topKeywords": []
                    },
                    "documents": [],
                    "slackDiscussions": []
                }

            # Encode query and perform vector search
            log_step("VECTOR_SEARCH_START", f"encoding query: '{query}'")
            query_vector = model.encode([query]).astype('float32')
            k = min(10, len(all_text_chunks))
            log_step("VECTOR_SEARCH_EXECUTE", f"searching with k={k}, total_chunks={len(all_text_chunks)}")
            D, I = index.search(query_vector, k=k)
            log_step("VECTOR_SEARCH_COMPLETE", f"found {len(I[0])} results, distances={D[0][:3].tolist()}")

            matched_texts = [all_text_chunks[i] for i in I[0] if 0 <= i < len(all_text_chunks)]
            log_step("MATCHED_TEXTS_EXTRACTED", f"valid_matches={len(matched_texts)}, total_text_length={sum(len(t) for t in matched_texts)}")

            matched_text_combined = "\n\n".join(matched_texts[:3])
            log_step("SUMMARY_GENERATION_START", f"combining top {min(3, len(matched_texts))} matches")
            summary = generate_summary(query, context, matched_text_combined)

            def match_metadata(data, query_vector, top_k=5):
                metadata_vectors = [
                    model.encode(f"{item.get('title', '')} {item.get('description', '')}").astype('float32')
                    for item in data
                ]
                metadata_vectors = np.array(metadata_vectors)
                query_vector = np.array(query_vector).reshape(1, -1)
                similarities = cosine_similarity(query_vector, metadata_vectors)[0]
                top_indices = np.argsort(similarities)[::-1][:top_k]
                return [data[idx] for idx in top_indices]

            log_step("METADATA_MATCHING_START", "searching metadata for keyword matches")
            matched_docs = match_metadata(doc_metadata, query_vector)
            matched_slacks = match_metadata(slack_data, query_vector)
            log_step("METADATA_MATCHING_COMPLETE", f"matched_docs={len(matched_docs)}, matched_slacks={len(matched_slacks)}")

            response = {
                "summary": {
                    "text": summary,
                    "matchCount": {
                        "total": len(matched_slacks) + len(matched_docs),
                        "slack": len(matched_slacks),
                        "docs": len(matched_docs),
                    },
                    "topKeywords": extract_top_keywords(matched_text_combined)
                },
                "documents": matched_docs,
                "slackDiscussions": matched_slacks
            }

            log_step("API_SEARCH_SUCCESS", f"response_summary_length={len(summary)}, total_results={len(matched_texts) + len(matched_docs) + len(matched_slacks)}")
            return response

    except Exception as e:
        log_step("API_SEARCH_ERROR", f"critical_error={str(e)}", "ERROR")
        return {"error": f"Search failed: {str(e)}"}


# --- MAIN ---
if __name__ == "__main__":
  log_step("APPLICATION_START", "RAG API application starting")
  
  log_step("DIRECTORY_SETUP_START", f"ensuring data directory exists: {DATA_DIR}")
  os.makedirs(DATA_DIR, exist_ok=True)
  log_step("DIRECTORY_SETUP_COMPLETE", f"data directory ready: {DATA_DIR}")
  
  log_step("INITIAL_DATA_LOAD_START", "performing initial data load")
  load_all_data()
  log_step("INITIAL_DATA_LOAD_COMPLETE", "initial data load finished")
  
  log_step("FILE_WATCHER_THREAD_START", "starting file watcher in background thread")
  threading.Thread(target=start_file_watcher, daemon=True).start()
  log_step("FILE_WATCHER_THREAD_STARTED", "file watcher thread started")
  
  log_step("FASTAPI_SERVER_START", "starting FastAPI server on port 8000")
  try:
      uvicorn.run("main:app", port=8000, reload=False)
      log_step("FASTAPI_SERVER_STARTED", "FastAPI server running successfully")
  except Exception as e:
      log_step("FASTAPI_SERVER_ERROR", f"server_error={str(e)}", "ERROR")
      raise e
  finally:
      log_step("APPLICATION_SHUTDOWN", "RAG API application shutting down")
