#!/usr/bin/env python3
"""
Flask Web Server for Explainer Video Generator
Provides a web interface for the video generation service.
"""

import os
import sys
import logging
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import threading
import uuid

# Import the video generator
from explainer_video import ExplainerVideoGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'dev-secret-key-change-in-production'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max request size

# Storage for job status
jobs = {}

# Initialize generator (will be done lazily)
generator = None


def get_generator():
    """Get or initialize the video generator."""
    global generator
    if generator is None:
        logger.info("Initializing video generator...")
        generator = ExplainerVideoGenerator(output_dir="outputs")
    return generator


def generate_video_async(job_id: str, prompt: str, title: str = None):
    """Generate video asynchronously in a background thread."""
    try:
        jobs[job_id]['status'] = 'processing'
        jobs[job_id]['message'] = 'Initializing...'
        
        gen = get_generator()
        
        # Update status
        jobs[job_id]['message'] = 'Generating video...'
        video_path = gen.generate(prompt, title=title)
        
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['video_path'] = video_path
        jobs[job_id]['message'] = 'Video generated successfully!'
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['message'] = str(e)


@app.route('/')
def index():
    """Serve the main web interface."""
    return render_template('index.html')


@app.route('/api/generate', methods=['POST'])
def generate():
    """
    API endpoint to generate a video.
    Expects JSON with 'prompt' and optional 'title'.
    """
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({'error': 'Prompt is required'}), 400
        
        prompt = data['prompt'].strip()
        title = data.get('title', None)
        
        # Validate prompt
        if not prompt:
            return jsonify({'error': 'Prompt cannot be empty'}), 400
        
        word_count = len(prompt.split())
        if word_count > 50:
            return jsonify({
                'warning': f'Prompt has {word_count} words. Recommended: <50 words for best results.'
            }), 400
        
        # Create job
        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            'id': job_id,
            'status': 'queued',
            'message': 'Job queued',
            'video_path': None
        }
        
        # Start generation in background thread
        thread = threading.Thread(
            target=generate_video_async,
            args=(job_id, prompt, title)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'status': 'queued',
            'message': 'Video generation started'
        }), 202
        
    except Exception as e:
        logger.error(f"Generation request failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """Get the status of a video generation job."""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    response = {
        'job_id': job_id,
        'status': job['status'],
        'message': job['message']
    }
    
    if job['status'] == 'completed' and job['video_path']:
        response['download_url'] = f"/api/download/{job_id}"
    
    return jsonify(response)


@app.route('/api/download/<job_id>', methods=['GET'])
def download_video(job_id):
    """Download the generated video."""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    
    if job['status'] != 'completed' or not job['video_path']:
        return jsonify({'error': 'Video not ready'}), 400
    
    try:
        return send_file(
            job['video_path'],
            mimetype='video/mp4',
            as_attachment=True,
            download_name='explainer_video.mp4'
        )
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return jsonify({'error': 'Failed to download video'}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'explainer-video-generator'})


if __name__ == '__main__':
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Flask server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)
