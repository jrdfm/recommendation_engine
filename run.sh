#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Navigate to webapp directory
cd webapp

# Run the FastAPI application
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Deactivate virtual environment when app is stopped
deactivate 