"""
AI Content Generation Agent
"""
import asyncio
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import openai
import whisper
import tempfile
from moviepy.editor import VideoFileClip
from clipfarming.agents import BaseAgent


class ContentGenerationAgent(BaseAgent):
    """Agent responsible for generating titles, descriptions, and subtitles using AI"""
    
    def __init__(self, config):
        super().__init__("ContentGenerationAgent", config)
        self.openai_client: Optional[openai.AsyncOpenAI] = None
        self.whisper_model = None
    
    async def setup(self):
        """Initialize AI models and clients"""
        await super().setup()
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            self.logger.warning("OpenAI API key not found. Some features may not work.")
        else:
            self.openai_client = openai.AsyncOpenAI(api_key=api_key)
        
        # Load Whisper model for transcription
        try:
            loop = asyncio.get_event_loop()
            self.whisper_model = await loop.run_in_executor(
                None, whisper.load_model, self.config.ai.whisper_model
            )
            self.logger.info(f"Loaded Whisper model: {self.config.ai.whisper_model}")
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {e}")
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method - generates content for clips
        
        Args:
            input_data: Dictionary containing clips information
            
        Returns:
            Dictionary containing clips with generated content
        """
        clips = input_data.get('clips', [])
        
        self.logger.info(f"Generating content for {len(clips)} clips")
        
        # Process clips concurrently
        semaphore = asyncio.Semaphore(3)  # Limit concurrent AI requests
        
        async def process_clip(clip_info):
            async with semaphore:
                return await self._generate_clip_content(clip_info)
        
        tasks = [process_clip(clip) for clip in clips]
        enhanced_clips = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        valid_clips = []
        for result in enhanced_clips:
            if isinstance(result, dict):
                valid_clips.append(result)
            elif isinstance(result, Exception):
                self.logger.error(f"Error generating content: {result}")
        
        return {
            'enhanced_clips': valid_clips,
            'original_clips': clips
        }
    
    async def _generate_clip_content(self, clip_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate title, description, and subtitles for a single clip
        
        Args:
            clip_info: Information about the clip
            
        Returns:
            Enhanced clip information with generated content
        """
        try:
            self.logger.info(f"Processing clip: {clip_info['title']}")
            
            # Make a copy of the clip info to avoid modifying original
            enhanced_clip = clip_info.copy()
            
            # Generate transcript
            transcript = await self._generate_transcript(clip_info['filename'])
            enhanced_clip['transcript'] = transcript
            
            # Generate title and description using the transcript
            if transcript and self.openai_client:
                title = await self._generate_title(transcript, clip_info)
                description = await self._generate_description(transcript, clip_info)
                hashtags = await self._generate_hashtags(transcript, clip_info)
                
                enhanced_clip['generated_title'] = title
                enhanced_clip['generated_description'] = description
                enhanced_clip['generated_hashtags'] = hashtags
            
            # Generate subtitles if enabled
            if self.config.editing.add_subtitles and transcript:
                subtitles_file = await self._generate_subtitles(
                    clip_info['filename'], transcript
                )
                enhanced_clip['subtitles_file'] = subtitles_file
            
            return enhanced_clip
            
        except Exception as e:
            self.logger.error(f"Error generating content for clip {clip_info['title']}: {e}")
            return clip_info
    
    async def _generate_transcript(self, video_path: str) -> str:
        """
        Generate transcript using Whisper
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Transcript text
        """
        if not self.whisper_model:
            self.logger.warning("Whisper model not available")
            return ""
        
        try:
            self.logger.info(f"Transcribing audio from: {video_path}")
            
            # Extract audio from video
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
            
            try:
                # Extract audio using moviepy
                loop = asyncio.get_event_loop()
                
                def extract_audio():
                    video = VideoFileClip(video_path)
                    audio = video.audio
                    audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
                    video.close()
                    audio.close()
                
                await loop.run_in_executor(None, extract_audio)
                
                # Transcribe using Whisper
                def transcribe():
                    result = self.whisper_model.transcribe(temp_audio_path)
                    return result["text"].strip()
                
                transcript = await loop.run_in_executor(None, transcribe)
                
                self.logger.info("Transcription completed")
                return transcript
                
            finally:
                # Clean up temporary audio file
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
                    
        except Exception as e:
            self.logger.error(f"Error transcribing video {video_path}: {e}")
            return ""
    
    async def _generate_title(self, transcript: str, clip_info: Dict[str, Any]) -> str:
        """
        Generate an engaging title using GPT
        
        Args:
            transcript: Video transcript
            clip_info: Clip metadata
            
        Returns:
            Generated title
        """
        try:
            # Extract key topic from transcript
            topic = await self._extract_topic(transcript)
            
            prompt = self.config.ai.title_prompt.format(topic=topic)
            
            response = await self.openai_client.chat.completions.create(
                model=self.config.ai.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a social media expert. Create engaging, click-worthy titles for video clips. Keep titles under 60 characters and make them attention-grabbing."
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}\n\nTranscript excerpt: {transcript[:500]}..."
                    }
                ],
                max_tokens=50,
                temperature=0.7
            )
            
            title = response.choices[0].message.content.strip()
            
            # Remove quotes if present
            if title.startswith('"') and title.endswith('"'):
                title = title[1:-1]
            
            self.logger.info(f"Generated title: {title}")
            return title
            
        except Exception as e:
            self.logger.error(f"Error generating title: {e}")
            return clip_info['title']  # Fallback to original title
    
    async def _generate_description(self, transcript: str, clip_info: Dict[str, Any]) -> str:
        """
        Generate a compelling description using GPT
        
        Args:
            transcript: Video transcript
            clip_info: Clip metadata
            
        Returns:
            Generated description
        """
        try:
            duration = int(clip_info['duration'])
            prompt = self.config.ai.description_prompt.format(duration=duration)
            
            response = await self.openai_client.chat.completions.create(
                model=self.config.ai.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a social media content creator. Write engaging descriptions for video clips that encourage views and engagement. Keep descriptions concise but compelling."
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}\n\nTranscript: {transcript[:1000]}..."
                    }
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            description = response.choices[0].message.content.strip()
            
            self.logger.info("Generated description")
            return description
            
        except Exception as e:
            self.logger.error(f"Error generating description: {e}")
            return f"Interesting clip from {clip_info['source_video']}"
    
    async def _generate_hashtags(self, transcript: str, clip_info: Dict[str, Any]) -> List[str]:
        """
        Generate relevant hashtags using GPT
        
        Args:
            transcript: Video transcript
            clip_info: Clip metadata
            
        Returns:
            List of hashtags
        """
        try:
            response = await self.openai_client.chat.completions.create(
                model=self.config.ai.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a hashtag expert. Generate 5-10 relevant hashtags for video content. Return only hashtags separated by spaces, each starting with #."
                    },
                    {
                        "role": "user",
                        "content": f"Generate hashtags for this video content:\n\n{transcript[:500]}..."
                    }
                ],
                max_tokens=100,
                temperature=0.5
            )
            
            hashtags_text = response.choices[0].message.content.strip()
            hashtags = [tag.strip() for tag in hashtags_text.split() if tag.startswith('#')]
            
            self.logger.info(f"Generated hashtags: {hashtags}")
            return hashtags
            
        except Exception as e:
            self.logger.error(f"Error generating hashtags: {e}")
            return ["#viral", "#trending", "#video"]  # Fallback hashtags
    
    async def _extract_topic(self, transcript: str) -> str:
        """
        Extract the main topic from transcript
        
        Args:
            transcript: Full transcript text
            
        Returns:
            Main topic/subject
        """
        try:
            if not transcript:
                return "general content"
            
            # Simple extraction - get first few sentences
            sentences = transcript.split('. ')[:3]
            topic_text = '. '.join(sentences)
            
            # If too long, use first 100 characters
            if len(topic_text) > 100:
                topic_text = topic_text[:100] + "..."
            
            return topic_text if topic_text else "interesting content"
            
        except Exception:
            return "general content"
    
    async def _generate_subtitles(self, video_path: str, transcript: str) -> Optional[str]:
        """
        Generate subtitle file (SRT format)
        
        Args:
            video_path: Path to video file
            transcript: Video transcript
            
        Returns:
            Path to subtitle file or None
        """
        try:
            if not transcript:
                return None
            
            # Create subtitles directory
            subtitles_dir = Path(self.config.storage.output_dir) / "subtitles"
            subtitles_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate subtitle filename
            video_name = Path(video_path).stem
            subtitle_path = subtitles_dir / f"{video_name}.srt"
            
            # Simple subtitle generation - split transcript into chunks
            words = transcript.split()
            words_per_subtitle = 8  # Approximately 8 words per subtitle
            subtitle_duration = 3.0  # 3 seconds per subtitle
            
            subtitles = []
            subtitle_index = 1
            
            for i in range(0, len(words), words_per_subtitle):
                start_time = i * subtitle_duration / words_per_subtitle
                end_time = start_time + subtitle_duration
                
                subtitle_text = " ".join(words[i:i + words_per_subtitle])
                
                # Format time for SRT (HH:MM:SS,mmm)
                start_srt = self._format_srt_time(start_time)
                end_srt = self._format_srt_time(end_time)
                
                subtitles.append(f"{subtitle_index}\n{start_srt} --> {end_srt}\n{subtitle_text}\n")
                subtitle_index += 1
            
            # Write subtitle file
            with open(subtitle_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(subtitles))
            
            self.logger.info(f"Generated subtitles: {subtitle_path}")
            return str(subtitle_path)
            
        except Exception as e:
            self.logger.error(f"Error generating subtitles: {e}")
            return None
    
    def _format_srt_time(self, seconds: float) -> str:
        """Format time for SRT subtitle format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"