#!/bin/bash
# Script to run the TSAD Orchestra Streamlit UI

# Ensure we're in the correct directory
cd "$(dirname "$0")" || exit 1

# Activate the virtual environment if it exists
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

# Run Streamlit app
streamlit run src/ui.py
