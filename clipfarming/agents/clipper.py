"""
Video Clipping and Editing Agent
"""
import asyncio
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import moviepy.editor as mp
import librosa
import numpy as np
from clipfarming.agents import BaseAgent


class VideoClippingAgent(BaseAgent):
    """Agent responsible for creating and editing video clips from full videos"""
    
    def __init__(self, config):
        super().__init__("VideoClippingAgent", config)
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method - creates clips from downloaded videos
        
        Args:
            input_data: Dictionary containing downloaded video information
            
        Returns:
            Dictionary containing created clips information
        """
        downloaded_videos = input_data.get('downloaded_videos', [])
        
        self.logger.info(f"Processing {len(downloaded_videos)} videos for clipping")
        
        all_clips = []
        
        # Process videos concurrently
        semaphore = asyncio.Semaphore(self.config.performance.max_concurrent_clips)
        
        async def process_video(video_info):
            async with semaphore:
                return await self._create_clips_from_video(video_info)
        
        tasks = [process_video(video) for video in downloaded_videos]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_clips.extend(result)
            elif isinstance(result, Exception):
                self.logger.error(f"Error processing video: {result}")
        
        return {
            'clips': all_clips,
            'original_videos': downloaded_videos
        }
    
    async def _create_clips_from_video(self, video_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create multiple clips from a single video
        
        Args:
            video_info: Information about the source video
            
        Returns:
            List of created clips information
        """
        video_path = video_info['filename']
        
        if not os.path.exists(video_path):
            self.logger.error(f"Video file not found: {video_path}")
            return []
        
        try:
            self.logger.info(f"Creating clips from: {video_info['title']}")
            
            # Load video
            loop = asyncio.get_event_loop()
            video = await loop.run_in_executor(None, mp.VideoFileClip, video_path)
            
            # Find interesting segments
            segments = await self._find_interesting_segments(video, video_info)
            
            # Create clips from segments
            clips = []
            for i, (start_time, end_time, engagement_score) in enumerate(segments):
                clip_info = await self._create_single_clip(
                    video, video_info, start_time, end_time, i, engagement_score
                )
                if clip_info:
                    clips.append(clip_info)
            
            # Close video to free memory
            video.close()
            
            self.logger.info(f"Created {len(clips)} clips from {video_info['title']}")
            return clips
            
        except Exception as e:
            self.logger.error(f"Error creating clips from {video_path}: {e}")
            return []
    
    async def _find_interesting_segments(
        self, video: mp.VideoFileClip, video_info: Dict[str, Any]
    ) -> List[Tuple[float, float, float]]:
        """
        Find interesting segments in the video based on audio/visual analysis
        
        Args:
            video: The video clip object
            video_info: Video metadata
            
        Returns:
            List of tuples (start_time, end_time, engagement_score)
        """
        duration = video.duration
        clip_length = self.config.video.clip_length_seconds
        clips_per_video = self.config.video.clips_per_video
        
        self.logger.info(f"Analyzing video for interesting segments (duration: {duration}s)")
        
        # If video is shorter than desired clip length, use the whole video
        if duration <= clip_length:
            return [(0, duration, 1.0)]
        
        try:
            # Extract audio for analysis
            audio_array = await self._extract_audio_array(video)
            
            # Analyze audio features
            segments_scores = await self._analyze_audio_features(audio_array, duration)
            
            # Select top segments
            top_segments = self._select_top_segments(
                segments_scores, duration, clip_length, clips_per_video
            )
            
            return top_segments
            
        except Exception as e:
            self.logger.warning(f"Error in segment analysis, using fallback method: {e}")
            return self._fallback_segment_selection(duration, clip_length, clips_per_video)
    
    async def _extract_audio_array(self, video: mp.VideoFileClip) -> np.ndarray:
        """Extract audio as numpy array for analysis"""
        loop = asyncio.get_event_loop()
        
        def extract():
            audio = video.audio
            # Get audio array at 22kHz sample rate
            return audio.to_soundarray(fps=22050)
        
        return await loop.run_in_executor(None, extract)
    
    async def _analyze_audio_features(
        self, audio_array: np.ndarray, duration: float
    ) -> List[Tuple[float, float, float]]:
        """
        Analyze audio features to find engaging segments
        
        Args:
            audio_array: Audio data as numpy array
            duration: Video duration in seconds
            
        Returns:
            List of (start_time, end_time, engagement_score) tuples
        """
        loop = asyncio.get_event_loop()
        
        def analyze():
            try:
                # Convert stereo to mono if necessary
                if len(audio_array.shape) > 1:
                    audio_mono = np.mean(audio_array, axis=1)
                else:
                    audio_mono = audio_array
                
                sr = 22050  # Sample rate
                
                # Calculate various audio features
                
                # 1. Energy (RMS)
                hop_length = 512
                frame_length = 2048
                rms = librosa.feature.rms(
                    y=audio_mono, hop_length=hop_length, frame_length=frame_length
                )[0]
                
                # 2. Spectral centroid (brightness)
                spectral_centroid = librosa.feature.spectral_centroid(
                    y=audio_mono, sr=sr, hop_length=hop_length
                )[0]
                
                # 3. Zero crossing rate (speech indicator)
                zcr = librosa.feature.zero_crossing_rate(
                    y=audio_mono, hop_length=hop_length
                )[0]
                
                # 4. Tempo and beat tracking
                tempo, beats = librosa.beat.beat_track(y=audio_mono, sr=sr)
                
                # Convert frame indices to time
                times = librosa.frames_to_time(
                    range(len(rms)), sr=sr, hop_length=hop_length
                )
                
                # Combine features into engagement score
                engagement_scores = []
                window_size = int(self.config.video.clip_length_seconds)
                
                for i in range(0, len(times) - window_size * sr // hop_length, sr // hop_length):
                    start_idx = i
                    end_idx = min(i + window_size * sr // hop_length, len(rms))
                    
                    if end_idx <= start_idx:
                        break
                    
                    # Calculate features for this window
                    window_rms = np.mean(rms[start_idx:end_idx])
                    window_centroid = np.mean(spectral_centroid[start_idx:end_idx])
                    window_zcr = np.mean(zcr[start_idx:end_idx])
                    
                    # Normalize and combine features
                    rms_norm = min(window_rms * 100, 1.0)  # Normalize RMS
                    centroid_norm = min(window_centroid / 4000, 1.0)  # Normalize centroid
                    zcr_norm = min(window_zcr * 10, 1.0)  # Normalize ZCR
                    
                    # Calculate engagement score (weighted combination)
                    engagement_score = (
                        0.4 * rms_norm +  # Audio energy
                        0.3 * centroid_norm +  # Brightness
                        0.3 * zcr_norm  # Speech activity
                    )
                    
                    start_time = times[start_idx]
                    end_time = min(start_time + window_size, duration)
                    
                    engagement_scores.append((start_time, end_time, engagement_score))
                
                return engagement_scores
                
            except Exception as e:
                self.logger.warning(f"Audio analysis failed: {e}")
                return []
        
        return await loop.run_in_executor(None, analyze)
    
    def _select_top_segments(
        self, segments_scores: List[Tuple[float, float, float]], 
        duration: float, clip_length: int, clips_per_video: int
    ) -> List[Tuple[float, float, float]]:
        """Select the top scoring segments with no overlap"""
        
        if not segments_scores:
            return self._fallback_segment_selection(duration, clip_length, clips_per_video)
        
        # Filter by minimum engagement threshold
        min_threshold = self.config.video.min_engagement_threshold
        filtered_segments = [
            seg for seg in segments_scores if seg[2] >= min_threshold
        ]
        
        if not filtered_segments:
            # If no segments meet threshold, use top segments anyway
            filtered_segments = segments_scores
        
        # Sort by engagement score (descending)
        filtered_segments.sort(key=lambda x: x[2], reverse=True)
        
        # Select non-overlapping segments
        selected = []
        for segment in filtered_segments:
            start, end, score = segment
            
            # Check for overlap with already selected segments
            overlap = False
            for sel_start, sel_end, _ in selected:
                if (start < sel_end and end > sel_start):
                    overlap = True
                    break
            
            if not overlap:
                selected.append(segment)
                
                if len(selected) >= clips_per_video:
                    break
        
        return selected
    
    def _fallback_segment_selection(
        self, duration: float, clip_length: int, clips_per_video: int
    ) -> List[Tuple[float, float, float]]:
        """Fallback method for segment selection when analysis fails"""
        
        segments = []
        
        if duration <= clip_length:
            return [(0, duration, 1.0)]
        
        # Divide video into equally spaced segments
        segment_spacing = duration / (clips_per_video + 1)
        
        for i in range(clips_per_video):
            start_time = segment_spacing * (i + 1) - clip_length / 2
            start_time = max(0, start_time)
            end_time = min(start_time + clip_length, duration)
            
            if end_time - start_time >= clip_length * 0.8:  # At least 80% of desired length
                segments.append((start_time, end_time, 0.5))  # Default score
        
        return segments
    
    async def _create_single_clip(
        self, video: mp.VideoFileClip, video_info: Dict[str, Any],
        start_time: float, end_time: float, clip_index: int, engagement_score: float
    ) -> Optional[Dict[str, Any]]:
        """
        Create a single clip with editing enhancements
        
        Args:
            video: Source video
            video_info: Video metadata
            start_time: Clip start time
            end_time: Clip end time
            clip_index: Index of this clip
            engagement_score: Engagement score of the segment
            
        Returns:
            Clip information dictionary
        """
        try:
            # Create output directory
            output_dir = Path(self.config.storage.output_dir) / "clips"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate clip filename
            safe_title = "".join(c for c in video_info['title'] if c.isalnum() or c in (' ', '-', '_')).strip()
            clip_filename = f"{safe_title}_clip_{clip_index:02d}.{self.config.video.output_format}"
            clip_path = output_dir / clip_filename
            
            self.logger.info(f"Creating clip: {clip_filename}")
            
            # Extract clip
            loop = asyncio.get_event_loop()
            
            def create_clip():
                # Extract the segment
                clip = video.subclip(start_time, end_time)
                
                # Apply editing enhancements
                if self.config.editing.remove_dead_air:
                    clip = self._remove_dead_air(clip)
                
                if self.config.editing.add_branding:
                    clip = self._add_branding(clip)
                
                # Set resolution if specified
                if self.config.video.output_resolution == "1080p":
                    clip = clip.resize(height=1080)
                elif self.config.video.output_resolution == "720p":
                    clip = clip.resize(height=720)
                
                # Write the clip
                clip.write_videofile(
                    str(clip_path),
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile='temp-audio.m4a',
                    remove_temp=True,
                    verbose=False,
                    logger=None
                )
                
                clip.close()
                return clip_path
            
            final_path = await loop.run_in_executor(None, create_clip)
            
            # Create clip metadata
            clip_info = {
                'filename': str(final_path),
                'title': f"{video_info['title']} - Clip {clip_index + 1}",
                'duration': end_time - start_time,
                'start_time': start_time,
                'end_time': end_time,
                'engagement_score': engagement_score,
                'source_video': video_info['title'],
                'source_url': video_info['url'],
                'clip_index': clip_index
            }
            
            self.logger.info(f"Successfully created clip: {clip_filename}")
            return clip_info
            
        except Exception as e:
            self.logger.error(f"Error creating clip {clip_index}: {e}")
            return None
    
    def _remove_dead_air(self, clip: mp.VideoFileClip) -> mp.VideoFileClip:
        """Remove silent segments from the clip"""
        # This is a simplified version - in a full implementation,
        # you would analyze audio to detect and remove silent segments
        return clip
    
    def _add_branding(self, clip: mp.VideoFileClip) -> mp.VideoFileClip:
        """Add branding overlay to the clip"""
        try:
            logo_path = Path(self.config.editing.branding_logo_path)
            if not logo_path.exists():
                return clip
            
            # Load logo
            logo = mp.ImageClip(str(logo_path)).set_duration(clip.duration)
            
            # Resize logo (make it small)
            logo = logo.resize(height=50)
            
            # Position logo based on config
            position = self.config.editing.branding_position
            if position == "bottom-right":
                logo = logo.set_position(('right', 'bottom')).set_margin(10)
            elif position == "top-right":
                logo = logo.set_position(('right', 'top')).set_margin(10)
            elif position == "bottom-left":
                logo = logo.set_position(('left', 'bottom')).set_margin(10)
            else:  # top-left
                logo = logo.set_position(('left', 'top')).set_margin(10)
            
            # Composite the logo over the clip
            return mp.CompositeVideoClip([clip, logo])
            
        except Exception as e:
            self.logger.warning(f"Could not add branding: {e}")
            return clip