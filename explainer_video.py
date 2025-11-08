#!/usr/bin/env python3
"""
Explainer Video Generator
Converts short text prompts (<50 words) into MP4 explainer videos.
Pipeline: parse key terms → summarize → voiceover → animate → merge audio/video.
Optimized for MacBook Pro M2 (8 GB RAM).
"""

import os
import sys
import logging
from typing import List, Tuple
from pathlib import Path
import tempfile

# NLP and ML
import spacy
from transformers import pipeline
import torch

# Text-to-Speech
from gtts import gTTS

# Animation
from manim import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExplainerVideoGenerator:
    """Main class for generating explainer videos from text prompts."""
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize the video generator.
        
        Args:
            output_dir: Directory to save output videos
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info("Initializing ExplainerVideoGenerator...")
        
        # Initialize spaCy for key term extraction
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✓ spaCy model loaded")
        except OSError:
            logger.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            raise
        
        # Initialize BART summarizer (optimized for low memory)
        try:
            device = 0 if torch.cuda.is_available() else -1
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=device,
                torch_dtype=torch.float16 if device == 0 else torch.float32
            )
            logger.info(f"✓ BART summarizer loaded (device: {'GPU' if device == 0 else 'CPU'})")
        except Exception as e:
            logger.error(f"Failed to load BART model: {e}")
            raise
    
    def parse_key_terms(self, text: str) -> List[str]:
        """
        Extract key terms from text using spaCy.
        
        Args:
            text: Input text
            
        Returns:
            List of key terms (nouns, proper nouns, and important entities)
        """
        logger.info("Parsing key terms...")
        doc = self.nlp(text)
        
        # Extract nouns, proper nouns, and named entities
        key_terms = []
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop:
                key_terms.append(token.text)
        
        # Add named entities
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT', 'WORK_OF_ART']:
                key_terms.append(ent.text)
        
        # Remove duplicates while preserving order
        key_terms = list(dict.fromkeys(key_terms))
        logger.info(f"✓ Found {len(key_terms)} key terms: {key_terms[:5]}...")
        return key_terms
    
    def summarize_text(self, text: str, min_length: int = 20, max_length: int = 100) -> str:
        """
        Summarize text using BART model.
        
        Args:
            text: Input text
            min_length: Minimum summary length
            max_length: Maximum summary length
            
        Returns:
            Summarized text
        """
        logger.info("Summarizing text...")
        
        # If text is already short, return as is
        if len(text.split()) <= max_length:
            logger.info("✓ Text already within target length")
            return text
        
        try:
            summary = self.summarizer(
                text,
                min_length=min_length,
                max_length=max_length,
                do_sample=False
            )[0]['summary_text']
            logger.info(f"✓ Summary generated ({len(summary.split())} words)")
            return summary
        except Exception as e:
            logger.error(f"Summarization failed: {e}. Using original text.")
            return text[:500]  # Fallback: truncate to reasonable length
    
    def generate_voiceover(self, text: str, output_path: str) -> str:
        """
        Generate voiceover audio using gTTS.
        
        Args:
            text: Text to convert to speech
            output_path: Path to save audio file
            
        Returns:
            Path to generated audio file
        """
        logger.info("Generating voiceover...")
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(output_path)
            logger.info(f"✓ Voiceover saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Voiceover generation failed: {e}")
            raise
    
    def create_animation(self, title: str, content: str, key_terms: List[str], 
                        output_path: str) -> str:
        """
        Create animation using Manim.
        
        Args:
            title: Video title
            content: Main content text
            key_terms: Key terms to highlight
            output_path: Path to save video file
            
        Returns:
            Path to generated video file
        """
        logger.info("Creating animation...")
        
        # Create a temporary Manim scene file
        scene_code = self._generate_scene_code(title, content, key_terms)
        temp_scene_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False, dir=str(self.output_dir)
        )
        temp_scene_file.write(scene_code)
        temp_scene_file.close()
        
        try:
            # Render with low quality for optimization
            import subprocess
            cmd = ["manim", "-ql", "--media_dir", str(self.output_dir), temp_scene_file.name, "ExplainerScene"]
            logger.info(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
            # Find the generated video
            video_path = self._find_latest_video(self.output_dir)
            logger.info(f"✓ Animation created: {video_path}")
            return video_path
        finally:
            # Clean up temp file
            os.unlink(temp_scene_file.name)
    
    def _generate_scene_code(self, title: str, content: str, key_terms: List[str]) -> str:
        """Generate Manim scene code dynamically."""
        # Escape strings for Python code
        title_escaped = title.replace('"', '\\"')
        content_escaped = content.replace('"', '\\"')
        
        # Wrap content text for better display
        content_lines = self._wrap_text(content, 50)
        
        scene_code = f'''
from manim import *

class ExplainerScene(Scene):
    def construct(self):
        # Title
        title = Text("{title_escaped}", font_size=48, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Content
        content_lines = {repr(content_lines)}
        content_group = VGroup()
        for line in content_lines:
            text = Text(line, font_size=24)
            content_group.add(text)
        content_group.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        content_group.next_to(title, DOWN, buff=0.5)
        
        self.play(FadeIn(content_group))
        self.wait(2)
        
        # Highlight key terms
        key_terms = {repr(key_terms[:3])}  # Limit to top 3
        if key_terms:
            highlights = VGroup()
            for i, term in enumerate(key_terms):
                highlight = Text(term, font_size=32, color=YELLOW)
                highlights.add(highlight)
            highlights.arrange(RIGHT, buff=0.5)
            highlights.to_edge(DOWN, buff=1)
            
            self.play(FadeIn(highlights))
            self.wait(1)
            self.play(FadeOut(highlights))
        
        self.wait(1)
'''
        return scene_code
    
    def _wrap_text(self, text: str, width: int) -> List[str]:
        """Wrap text to specified width."""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def _find_latest_video(self, search_dir: Path) -> str:
        """Find the most recently created video file."""
        video_files = list(search_dir.rglob("*.mp4"))
        if not video_files:
            raise FileNotFoundError("No video file found after rendering")
        latest = max(video_files, key=lambda p: p.stat().st_mtime)
        return str(latest)
    
    def merge_audio_video(self, video_path: str, audio_path: str, output_path: str) -> str:
        """
        Merge audio and video using FFmpeg.
        
        Args:
            video_path: Path to video file
            audio_path: Path to audio file
            output_path: Path to save merged file
            
        Returns:
            Path to merged video file
        """
        logger.info("Merging audio and video...")
        import subprocess
        cmd = ["ffmpeg", "-i", video_path, "-i", audio_path, "-c:v", "copy", "-c:a", "aac", "-shortest", "-y", output_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(output_path):
            logger.info(f"✓ Final video saved to {output_path}")
            return output_path
        else:
            logger.error(f"FFmpeg stderr: {result.stderr}")
            raise RuntimeError(f"Failed to merge audio and video: {result.stderr}")
    
    def generate(self, prompt: str, title: str = None) -> str:
        """
        Generate complete explainer video from prompt.
        
        Args:
            prompt: Input text prompt (<50 words)
            title: Optional title for video (defaults to first few words of prompt)
            
        Returns:
            Path to generated video file
        """
        logger.info(f"Starting video generation for prompt: '{prompt[:50]}...'")
        
        # Validate prompt length
        word_count = len(prompt.split())
        if word_count > 50:
            logger.warning(f"Prompt has {word_count} words (>50). Consider shortening.")
        
        # Generate title if not provided
        if title is None:
            title = ' '.join(prompt.split()[:5]) + "..."
        
        try:
            # Step 1: Parse key terms
            key_terms = self.parse_key_terms(prompt)
            
            # Step 2: Summarize (if needed)
            summary = self.summarize_text(prompt)
            
            # Step 3: Generate voiceover
            audio_path = str(self.output_dir / "voiceover.mp3")
            self.generate_voiceover(summary, audio_path)
            
            # Step 4: Create animation
            video_path = self.create_animation(title, summary, key_terms, str(self.output_dir))
            
            # Step 5: Merge audio and video
            final_output = str(self.output_dir / "explainer_video.mp4")
            self.merge_audio_video(video_path, audio_path, final_output)
            
            logger.info(f"✅ Video generation complete: {final_output}")
            return final_output
            
        except Exception as e:
            logger.error(f"❌ Video generation failed: {e}")
            raise


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate explainer videos from text prompts"
    )
    parser.add_argument(
        "prompt",
        type=str,
        help="Text prompt to convert to video (<50 words recommended)"
    )
    parser.add_argument(
        "-t", "--title",
        type=str,
        default=None,
        help="Video title (optional)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="outputs",
        help="Output directory (default: outputs)"
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        generator = ExplainerVideoGenerator(output_dir=args.output)
        video_path = generator.generate(args.prompt, title=args.title)
        print(f"\n✅ Success! Video saved to: {video_path}")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
