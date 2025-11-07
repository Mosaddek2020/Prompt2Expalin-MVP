# Prompt2Explain-MVP

🎬 **AI-Powered Explainer Video Generator**

Prompt2Explain is an AI-powered MVP that converts short text prompts (<50 words) on math/science topics into engaging MP4 explainer videos. Designed for educational use, it generates visual content like a teacher using a whiteboard, helping users quickly grasp complex topics through engaging and dynamic presentations.

## ✨ Features

- **🧠 Smart Text Analysis**: Uses spaCy to extract key terms from your prompt
- **📝 AI Summarization**: Employs BART (Transformers) to create concise summaries (20-100 words)
- **🎙️ Natural Voiceover**: Generates audio narration using Google Text-to-Speech (gTTS)
- **🎨 Dynamic Animations**: Creates beautiful animations with Manim (titles, text, highlights, MathTex)
- **🎬 Professional Output**: Merges audio and video using FFmpeg into polished MP4 files
- **⚡ Optimized Performance**: Configured for MacBook Pro M2 (8 GB RAM) with low-res rendering
- **🌐 Web Interface**: Modern, responsive frontend for easy video generation
- **🔄 Real-time Status**: Live progress updates during video generation

## 🏗️ Architecture

### Pipeline
```
Text Prompt → Parse Key Terms → Summarize → Generate Voiceover → Animate → Merge A/V → MP4 Output
```

### Components
1. **Backend (`explainer_video.py`)**: Core video generation engine (~200 lines)
2. **Web Server (`app.py`)**: Flask API server with async job processing
3. **Frontend**: Modern HTML/CSS/JS interface with real-time updates

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **FFmpeg** (for video processing)
- **8 GB RAM** minimum (optimized for MacBook Pro M2)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Mosaddek2020/Prompt2Expalin-MVP.git
   cd Prompt2Expalin-MVP
   ```

2. **Install FFmpeg** (macOS):
   ```bash
   brew install ffmpeg
   ```
   
   For other platforms, see [FFmpeg installation guide](https://ffmpeg.org/download.html).

3. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Download spaCy language model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

### Usage

#### Option 1: Web Interface (Recommended)

1. **Start the web server**:
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

3. **Enter your prompt** and click "Generate Video"!

#### Option 2: Command Line

```bash
python explainer_video.py "Your prompt here" -t "Optional Title" -o outputs
```

**Example**:
```bash
python explainer_video.py "Explain the Pythagorean theorem and its applications in real life" -t "Pythagorean Theorem"
```

**Options**:
- `-t, --title`: Custom video title (optional)
- `-o, --output`: Output directory (default: outputs)
- `-d, --debug`: Enable debug logging

## 📁 Project Structure

```
Prompt2Expalin-MVP/
├── explainer_video.py      # Core video generation engine
├── app.py                  # Flask web server
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore rules
├── README.md              # This file
├── templates/             # HTML templates
│   └── index.html         # Main web interface
├── static/                # Static assets
│   ├── css/
│   │   └── style.css      # Stylesheet
│   └── js/
│       └── app.js         # Frontend JavaScript
└── outputs/               # Generated videos (created automatically)
```

## 🎯 API Reference

### Web API Endpoints

#### Generate Video
```http
POST /api/generate
Content-Type: application/json

{
  "prompt": "Your text prompt here",
  "title": "Optional title"
}

Response:
{
  "job_id": "uuid",
  "status": "queued",
  "message": "Video generation started"
}
```

#### Check Status
```http
GET /api/status/{job_id}

Response:
{
  "job_id": "uuid",
  "status": "completed|processing|failed",
  "message": "Status message",
  "download_url": "/api/download/{job_id}"  // if completed
}
```

#### Download Video
```http
GET /api/download/{job_id}

Response: video/mp4 file
```

## 🔧 Configuration

### Performance Optimization

The system is optimized for MacBook Pro M2 with 8 GB RAM:

- **Low-resolution rendering** (`-ql` flag in Manim)
- **Float16 precision** for transformer models (when GPU available)
- **Efficient memory management** with streaming and cleanup
- **Async processing** to prevent UI blocking

### Environment Variables

Create a `.env` file to customize settings:

```env
PORT=5000
DEBUG=False
MAX_PROMPT_WORDS=50
OUTPUT_DIR=outputs
```

## 🛠️ Development

### Debug Mode

Enable detailed logging:
```bash
python explainer_video.py "Your prompt" --debug
```

Or for the web server:
```bash
DEBUG=True python app.py
```

### Testing

Test the pipeline with a simple prompt:
```bash
python explainer_video.py "The sun is a star that provides light and heat to Earth"
```

Expected output: `outputs/explainer_video.mp4`

## 📝 Examples

### Example 1: Math Concept
```
Prompt: "The Pythagorean theorem states that in a right triangle, 
         a² + b² = c², where c is the hypotenuse"
Result: Video with theorem explanation, formula animation, and voiceover
```

### Example 2: Science Concept
```
Prompt: "Photosynthesis is how plants convert sunlight, water, and 
         carbon dioxide into oxygen and glucose"
Result: Video explaining the process with key terms highlighted
```

## 🐛 Troubleshooting

### Common Issues

1. **"spaCy model not found"**
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **"FFmpeg not found"**
   - Install FFmpeg: `brew install ffmpeg` (macOS)
   - Ensure FFmpeg is in your PATH

3. **"Out of memory"**
   - Reduce prompt length (<50 words)
   - Close other applications
   - System is optimized for 8GB RAM

4. **"Manim rendering failed"**
   - Check logs in `outputs/` directory
   - Ensure sufficient disk space
   - Try with a simpler prompt first

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

Built with:
- [spaCy](https://spacy.io/) - NLP and key term extraction
- [Transformers](https://huggingface.co/transformers/) - BART summarization
- [gTTS](https://github.com/pndurette/gTTS) - Text-to-speech
- [Manim](https://www.manim.community/) - Mathematical animations
- [FFmpeg](https://ffmpeg.org/) - Video processing
- [Flask](https://flask.palletsprojects.com/) - Web framework

## 📧 Support

For issues and questions, please open an issue on GitHub.

---

Made with ❤️ for education
