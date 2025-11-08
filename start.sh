#!/bin/bash
# Quick start script for Explainer Video Generator

echo "🎬 Starting Prompt→Explainer Video Generator..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run the following commands first:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    echo "  python -m spacy download en_core_web_sm"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if spaCy model is installed
python -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📥 Installing spaCy language model..."
    python -m spacy download en_core_web_sm
fi

# Create outputs directory
mkdir -p outputs

# Start the server
echo ""
echo "✅ Starting web server..."
echo "🌐 Open http://localhost:5000 in your browser"
echo ""
python app.py
