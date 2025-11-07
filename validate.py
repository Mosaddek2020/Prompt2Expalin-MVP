#!/usr/bin/env python3
"""
Test script to validate the Explainer Video Generator structure
This script checks that all required components are in place without running the full pipeline.
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and report."""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description} missing: {filepath}")
        return False

def check_imports(filepath, required_imports):
    """Check if a Python file contains required imports."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        missing = []
        for imp in required_imports:
            if imp not in content:
                missing.append(imp)
        
        if not missing:
            print(f"✅ All required imports present in {filepath}")
            return True
        else:
            print(f"⚠️  Missing imports in {filepath}: {missing}")
            return False
    except Exception as e:
        print(f"❌ Error checking imports in {filepath}: {e}")
        return False

def main():
    """Run validation checks."""
    print("🔍 Validating Prompt→Explainer Video Generator Structure\n")
    
    all_passed = True
    
    # Check core files
    print("📁 Checking Core Files:")
    all_passed &= check_file_exists("explainer_video.py", "Backend script")
    all_passed &= check_file_exists("app.py", "Web server")
    all_passed &= check_file_exists("requirements.txt", "Dependencies")
    all_passed &= check_file_exists("README.md", "Documentation")
    all_passed &= check_file_exists(".gitignore", "Git ignore")
    all_passed &= check_file_exists("start.sh", "Startup script")
    print()
    
    # Check frontend files
    print("🌐 Checking Frontend Files:")
    all_passed &= check_file_exists("templates/index.html", "HTML template")
    all_passed &= check_file_exists("static/css/style.css", "Stylesheet")
    all_passed &= check_file_exists("static/js/app.js", "JavaScript")
    print()
    
    # Check required imports in backend
    print("📦 Checking Backend Dependencies:")
    backend_imports = [
        "import spacy",
        "from transformers import",
        "from gtts import gTTS",
        "from manim import",
        "import logging"
    ]
    all_passed &= check_imports("explainer_video.py", backend_imports)
    print()
    
    # Check required imports in web server
    print("🌐 Checking Web Server Dependencies:")
    server_imports = [
        "from flask import Flask",
        "from flask_cors import CORS",
        "import threading"
    ]
    all_passed &= check_imports("app.py", server_imports)
    print()
    
    # Check key functions in backend
    print("🔧 Checking Backend Functions:")
    try:
        with open("explainer_video.py", 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print("❌ explainer_video.py not found")
        return False
    except Exception as e:
        print(f"❌ Error reading explainer_video.py: {e}")
        return False
    
    required_functions = [
        "def parse_key_terms",
        "def summarize_text",
        "def generate_voiceover",
        "def create_animation",
        "def merge_audio_video",
        "def generate"
    ]
    
    for func in required_functions:
        if func in content:
            print(f"✅ {func} implemented")
        else:
            print(f"❌ {func} missing")
            all_passed = False
    print()
    
    # Check Flask routes
    print("🛣️  Checking API Routes:")
    try:
        with open("app.py", 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print("❌ app.py not found")
        return False
    except Exception as e:
        print(f"❌ Error reading app.py: {e}")
        return False
    
    required_routes = [
        "@app.route('/')",
        "@app.route('/api/generate'",
        "@app.route('/api/status",
        "@app.route('/api/download"
    ]
    
    for route in required_routes:
        if route in content:
            print(f"✅ {route} implemented")
        else:
            print(f"❌ {route} missing")
            all_passed = False
    print()
    
    # Check requirements.txt
    print("📋 Checking Requirements:")
    try:
        with open("requirements.txt", 'r') as f:
            requirements = f.read()
    except FileNotFoundError:
        print("❌ requirements.txt not found")
        return False
    except Exception as e:
        print(f"❌ Error reading requirements.txt: {e}")
        return False
    
    required_packages = [
        "spacy",
        "transformers",
        "torch",
        "gTTS",
        "manim",
        "Flask",
        "Flask-CORS"
    ]
    
    for package in required_packages:
        if package in requirements:
            print(f"✅ {package} in requirements.txt")
        else:
            print(f"❌ {package} missing from requirements.txt")
            all_passed = False
    print()
    
    # Final result
    print("=" * 50)
    if all_passed:
        print("✅ All validation checks passed!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Download spaCy model: python -m spacy download en_core_web_sm")
        print("3. Start server: python app.py")
        print("4. Open http://localhost:5000 in browser")
        return 0
    else:
        print("❌ Some validation checks failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
