# RAG-Powered Search API with Groq LLM Integration

## Overview
This project implements a Retrieval-Augmented Generation (RAG) search API that combines vector similarity search with LLM-powered summarization. It's designed to search through internal documentation, including text files and structured data (like Slack discussions and metadata), and provide relevant, contextualized responses.

## Features
- Real-time document indexing and search
- File system monitoring for automatic index updates
- Vector similarity search using FAISS
- LLM-powered response summarization using Groq
- Support for multiple data sources (text files, JSON metadata, Slack discussions)
- Automatic text chunking and embedding generation

## Architecture

### Components
1. **FastAPI Server**: Handles HTTP requests and API endpoints
2. **Vector Store**: FAISS index for similarity search
3. **Embedding Model**: Sentence Transformer (all-MiniLM-L6-v2)
4. **LLM Integration**: Groq API for response summarization
5. **File Watcher**: Monitors data directory for changes

### Data Processing Pipeline
1. Document Loading → Text Chunking → Embedding Generation → FAISS Indexing
2. Search Query → Vector Search → Context Retrieval → LLM Summarization

## API Endpoints

### POST /search
Searches through indexed documents and returns relevant matches with a summarized response.

#### Input JSON Structure
```json
{
    "query": "string",    // The search query
    "context": "string"   // Optional additional context for the search
}
```

#### Output JSON Structure
```json
{
    "summary": {
        "text": "string",           // LLM-generated summary
        "matchCount": {
            "total": int,           // Total number of matches
            "slack": int,           // Number of Slack discussion matches
            "docs": int,            // Number of document matches
            "resources": int        // Number of resource matches
        },
        "topKeywords": ["string"]   // List of key terms from the query
    },
    "documents": [                  // List of matching documents
        {
            "title": "string",
            "description": "string",
            "type": "string",
            "timeAgo": "string",
            "imageUrl": "string",
            "author": "string",
            "date": "string",
            "team": "string"
        }
    ],
    "slackDiscussions": [          // List of matching Slack discussions
        {
            "title": "string",
            "description": "string",
            "daysAgo": int,
            "replies": int,
            "author": "string",
            "date": "string",
            "channel": "string",
            "reactions": int
        }
    ],
    "resources": []                // List of additional resources (if any)
}
```

## Configuration

### Environment Variables and Constants
- `DATA_DIR`: "./data" - Directory for source documents
- `GROQ_API_KEY`: API key for Groq LLM service
- `GROQ_MODEL`: "llama3-70b-8192" - Groq model identifier
- `INDEX_FILE`: "faiss.index" - FAISS index file location
- `LOG_FILE`: "rag_api_flow.log" - Log file location

### Hyperparameters

#### Text Processing
- `chunk_size`: 512 characters - Size of text chunks for processing
- `overlap`: 50 characters - Overlap between consecutive chunks

#### Vector Search
- `nlist`: min(100, num_vectors // 10) - Number of clusters for IVF index
- `dimension`: 384 - Default embedding dimension for all-MiniLM-L6-v2 model
- `k`: min(10, len(all_text_chunks)) - Number of similar chunks to retrieve
- Index type selection:
  - Flat index for < 100 vectors (IndexFlatL2)
  - IVF index for >= 100 vectors (IndexIVFFlat)
- Distance metric: L2 (Euclidean distance)

#### LLM Configuration
- Model: llama3-70b-8192
- `max_tokens`: 1024 - Maximum tokens in LLM response
- Temperature: Default (not explicitly set)
- System role: "You summarize trading and performance data."

#### Search and Matching
- `top_matches_for_summary`: 3 - Number of top matches used for summary generation
- Metadata matching:
  - Case-insensitive keyword matching
  - Minimum keyword length: > 2 characters
  - Stop words excluded: ['the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']

#### System Configuration
- Server port: 8000
- File watcher: Recursive=False (only monitors top-level directory)
- Thread configuration: File watcher runs in daemon thread

## Data Directory Structure
```
./data/
├── documents.json      # Document metadata
├── slack_discussions.json  # Slack discussion data
└── *.txt              # Text documents for indexing
```

## Dependencies
- Python 3.7+
- FastAPI 0.104.1
- Uvicorn[standard] 0.24.0
- Sentence-Transformers 2.2.2
- FAISS-cpu 1.7.4
- NumPy 1.24.3
- Requests 2.31.0
- Watchdog 3.0.0
- Pydantic 2.5.0

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your data directory:
- Create a `data` directory
- Add your text files
- Create `documents.json` and `slack_discussions.json` with appropriate metadata

3. Start the server:
```bash
uvicorn main:app --reload
```

4. The API will be available at `http://localhost:8000`

## Logging
The application maintains detailed logs in `rag_api_flow.log`, tracking:
- Data loading and processing steps
- Search operations
- LLM interactions
- Error conditions

## Error Handling
- File reading errors are logged and skipped
- LLM API errors return error messages in the summary
- Index creation failures are handled gracefully
- File system monitoring errors are logged

## Performance Considerations
- Adaptive indexing strategy based on dataset size
- Efficient chunk management with hash-based deduplication
- Thread-safe data loading with mutex locks
- Automatic index updates on file changes

## Running with Docker (Recommended)

The easiest way to run the API is using Docker:

1. Build and start the container:
```bash
docker-compose up --build
```

2. The API will be available at http://localhost:8000

3. To stop the service:
```bash
docker-compose down
```

Note: The `data` directory and FAISS index are mounted as volumes, so your data and index persist between container restarts.

## Environment Setup (Without Docker)

If you prefer to run without Docker:

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create data directory:
```bash
mkdir -p data
```

3. Run the API:
```bash
python main.py
```

## API Usage

Search endpoint:
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "your search query", "context": ""}'
```

## Configuration

- FAISS vector dimension: 384
- Text chunk size: 512
- Chunk overlap: 50
- Model: sentence-transformers for embeddings
- Data directory: /app/data (Docker) or ./data (local) 