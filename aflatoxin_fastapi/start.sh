#!/bin/bash

# Start the Aflatoxin Risk Prediction API

echo "Starting Aflatoxin Risk Prediction API..."

# Check if running in Docker
if [ -f /.dockerenv ]; then
    echo "Running in Docker container"
    exec uvicorn app.main:app --host 0.0.0.0 --port 8000
else
    echo "Running locally"
    # Install dependencies if needed
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt
    else
        source venv/bin/activate
    fi
    
    # Start the server
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
fi