#!/bin/bash

# Ensure we're in the correct directory
cd "$(dirname "$0")"

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the server
python -m uvicorn main:app --host 0.0.0.0 --port $PORT 