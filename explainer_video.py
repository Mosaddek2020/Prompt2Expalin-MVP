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
        
        # Use lazy loading for heavy models so importing/instantiating is fast
        logger.info("Initializing ExplainerVideoGenerator (lazy model loading)...")
        self.nlp = None  # spaCy NLP (load on demand)
        self.summarizer = None  # HuggingFace summarizer (load on demand)
        # record device choice for possible summarizer loading later
        self.device = 0 if torch.cuda.is_available() else -1
        self.models_loaded = False

    def ensure_nlp_loaded(self):
        """Load spaCy model if not already loaded."""
        if self.nlp is None:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("✓ spaCy model loaded")
            except OSError:
                logger.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
                raise

    def ensure_summarizer_loaded(self):
        """Load the summarizer model on demand. This model is large (may download).

        Call this only when you actually need summarization to avoid long downloads.
        """
        if self.summarizer is None:
            try:
                self.summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=self.device,
                    torch_dtype=torch.float16 if self.device == 0 else torch.float32
                )
                logger.info(f"✓ BART summarizer loaded (device: {'GPU' if self.device == 0 else 'CPU'})")
                self.models_loaded = True
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
        # Ensure spaCy is loaded before parsing
        self.ensure_nlp_loaded()
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
        Process text for video content. If the text is a short prompt/question,
        generate a simple explanation. If it's long content, summarize it.
        
        Args:
            text: Input text (can be a prompt/question or full content)
            min_length: Minimum output length
            max_length: Maximum output length
            
        Returns:
            Processed text suitable for video narration
        """
        logger.info("Processing text for video...")
        word_count = len(text.split())
        
        # If text is a short prompt/question (< 15 words), create a detailed explanation
        if word_count < 15:
            logger.info("✓ Short prompt detected - creating detailed explanation")
            # Extract the main topic from the prompt
            self.ensure_nlp_loaded()
            doc = self.nlp(text)
            
            # Try to extract noun chunks (compound nouns like "black holes")
            noun_chunks = [chunk.text.lower() for chunk in doc.noun_chunks if not all(token.is_stop for token in chunk)]
            
            if noun_chunks:
                main_topic = noun_chunks[0]
                # Determine singular/plural for grammar
                is_plural = main_topic.endswith('s') or ' ' in main_topic
                verb = "are" if is_plural else "is"
                pronoun = "They" if is_plural else "It"
                
                # Generate a longer, more detailed explanation
                explanation = f"""{main_topic.capitalize()} {verb} fascinating concepts in science that have captivated researchers and enthusiasts alike. 
                
{pronoun} represent fundamental phenomena that help us understand the workings of our universe at both microscopic and cosmic scales. 

The study of {main_topic} involves multiple disciplines including physics, mathematics, and computational science. 

Understanding {main_topic} requires examining their properties, behaviors, and the principles that govern their existence. 

These concepts play crucial roles in advancing our knowledge and developing new technologies. 

Scientists continue to explore {main_topic} through observation, experimentation, and theoretical modeling. 

Let's dive deeper into the key aspects, important characteristics, and fascinating details about {main_topic} that make {pronoun.lower()} so significant in modern science."""
            else:
                # Fallback if no topics found
                explanation = f"""{text}. This is an intriguing and important topic in science and mathematics. 
                
It encompasses various concepts and principles that contribute to our understanding of the natural world. 

Exploring this subject reveals connections between different scientific disciplines and real-world applications. 

The study of this topic involves both theoretical frameworks and practical investigations. 

Researchers have made significant discoveries that continue to shape our knowledge in this area. 

Let's examine the fundamental concepts, key principles, and interesting facts that make this topic essential for scientific literacy."""
            
            logger.info(f"✓ Generated detailed explanation ({len(explanation.split())} words)")
            return explanation
        
        # If text is medium length, use as-is
        if word_count <= max_length:
            logger.info("✓ Text length is appropriate")
            return text
        
        # If text is long, summarize it
        try:
            logger.info("Long text detected - will summarize")
            # Load summarizer lazily (this may download a large model)
            self.ensure_summarizer_loaded()

            summary = self.summarizer(
                text,
                min_length=min_length,
                max_length=max_length,
                do_sample=False
            )[0]['summary_text']
            logger.info(f"✓ Summary generated ({len(summary.split())} words)")
            return summary
        except Exception as e:
            logger.error(f"Summarization failed: {e}. Using truncated text.")
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
        """Generate Manim scene code dynamically with enhanced animations."""
        # Escape strings for Python code
        title_escaped = title.replace('"', '\\"')
        content_escaped = content.replace('"', '\\"')
        
        # Split content into paragraphs for better pacing
        paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
        
        scene_code = f'''
from manim import *

class ExplainerScene(Scene):
    def construct(self):
        # Set background color
        self.camera.background_color = "#0a0e27"
        
        # Enhanced title sequence with underline
        title = Text("{title_escaped}", font_size=60, color=BLUE, weight=BOLD)
        title.to_edge(UP, buff=0.8)
        
        # Add decorative underline
        underline = Line(
            start=title.get_left() + DOWN*0.3,
            end=title.get_right() + DOWN*0.3,
            color=YELLOW,
            stroke_width=4
        )
        
        # Animate title with scale effect
        self.play(
            Write(title, run_time=2),
            Create(underline, run_time=1.5)
        )
        self.wait(1.5)
        
        # Process content in chunks for better pacing
        paragraphs = {repr(paragraphs)}
        
        for i, para in enumerate(paragraphs):
            # Clear previous content with smooth fade
            if i > 0:
                self.play(
                    FadeOut(content_text, shift=UP*0.5),
                    run_time=0.8
                )
                self.wait(0.3)
            
            # Wrap text to fit screen
            words = para.split()
            lines = []
            current_line = []
            max_width = 45
            
            for word in words:
                test_line = ' '.join(current_line + [word])
                if len(test_line) <= max_width:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [word]
            if current_line:
                lines.append(' '.join(current_line))
            
            # Create text group with color gradient
            content_group = VGroup()
            colors = [WHITE, "#e0e0e0", "#d0d0d0"]
            
            for idx, line in enumerate(lines):
                text_obj = Text(
                    line, 
                    font_size=26,
                    color=colors[min(idx % 3, len(colors)-1)]
                )
                content_group.add(text_obj)
            
            content_group.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
            content_group.next_to(title, DOWN, buff=1.2)
            content_text = content_group
            
            # Animate in with cascading effect
            for line in content_group:
                self.play(
                    FadeIn(line, shift=RIGHT*0.3),
                    run_time=0.4
                )
            
            # Add a subtle pulse effect
            self.play(
                content_text.animate.scale(1.02),
                run_time=0.3
            )
            self.play(
                content_text.animate.scale(1/1.02),
                run_time=0.3
            )
            
            # Reading time
            reading_time = max(2.5, len(words) / 3)
            self.wait(reading_time)
        
        # Enhanced key terms section with animations
        key_terms = {repr(key_terms[:5])}
        if key_terms:
            self.play(
                FadeOut(content_text, shift=DOWN*0.5),
                run_time=0.8
            )
            self.wait(0.5)
            
            # Key concepts title with circle decoration
            key_title = Text("Key Concepts", font_size=48, color=YELLOW, weight=BOLD)
            key_title.next_to(title, DOWN, buff=1.2)
            
            # Decorative circles around title
            circle1 = Circle(radius=0.3, color=YELLOW, stroke_width=3)
            circle2 = Circle(radius=0.3, color=YELLOW, stroke_width=3)
            circle1.next_to(key_title, LEFT, buff=0.4)
            circle2.next_to(key_title, RIGHT, buff=0.4)
            
            self.play(
                Write(key_title, run_time=1.5),
                Create(circle1),
                Create(circle2)
            )
            self.wait(0.8)
            
            # Animated key terms with icons
            terms_group = VGroup()
            bullets = VGroup()
            
            for i, term in enumerate(key_terms):
                # Create bullet point
                bullet = Dot(color=BLUE, radius=0.12)
                
                # Create term text
                term_text = Text(f"  {{term}}", font_size=32, color=WHITE)
                
                # Group bullet and text
                term_line = VGroup(bullet, term_text).arrange(RIGHT, buff=0.2)
                terms_group.add(term_line)
                bullets.add(bullet)
            
            terms_group.arrange(DOWN, aligned_edge=LEFT, buff=0.5)
            terms_group.next_to(key_title, DOWN, buff=1)
            
            # Animate each term with emphasis
            for i, term_line in enumerate(terms_group):
                bullet, text = term_line
                
                # Growing circle effect for bullet
                temp_circle = Circle(radius=0.12, color=BLUE).move_to(bullet)
                self.play(
                    GrowFromCenter(temp_circle),
                    run_time=0.4
                )
                self.play(
                    Transform(temp_circle, bullet),
                    run_time=0.3
                )
                self.remove(temp_circle)
                self.add(bullet)
                
                # Slide in text
                self.play(
                    Write(text, run_time=0.8)
                )
                
                # Brief highlight effect
                highlight_rect = SurroundingRectangle(
                    text,
                    color=YELLOW,
                    buff=0.15,
                    stroke_width=2
                )
                self.play(Create(highlight_rect), run_time=0.4)
                self.play(FadeOut(highlight_rect), run_time=0.4)
                
                self.wait(1)
            
            self.wait(2)
            
            # Fade out key concepts
            self.play(
                FadeOut(key_title),
                FadeOut(circle1),
                FadeOut(circle2),
                FadeOut(terms_group),
                run_time=1
            )
        
        # Enhanced outro with animation
        self.wait(0.5)
        
        # Create star decorations
        stars = VGroup()
        for _ in range(8):
            star = Star(n=5, outer_radius=0.15, color=YELLOW, fill_opacity=0.8)
            star.move_to([
                np.random.uniform(-6, 6),
                np.random.uniform(-3, 3),
                0
            ])
            stars.add(star)
        
        # Outro message
        outro = Text("Thank you for watching!", font_size=52, color=GREEN, weight=BOLD)
        
        # Animate stars twinkling
        self.play(
            LaggedStart(
                *[GrowFromCenter(star) for star in stars],
                lag_ratio=0.15
            ),
            run_time=1.5
        )
        
        # Zoom in outro text
        outro.scale(0.01)
        self.add(outro)
        self.play(
            outro.animate.scale(100),
            run_time=1.5,
            rate_func=smooth
        )
        
        # Pulse effect
        self.play(
            outro.animate.scale(1.1),
            stars.animate.set_opacity(1),
            run_time=0.4
        )
        self.play(
            outro.animate.scale(1/1.1),
            run_time=0.4
        )
        
        self.wait(2)
        
        # Final fadeout with rotation
        self.play(
            FadeOut(title, shift=UP),
            FadeOut(underline, shift=UP),
            FadeOut(outro, shift=DOWN),
            FadeOut(stars, scale=0.5),
            run_time=2
        )
        self.wait(0.5)
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
