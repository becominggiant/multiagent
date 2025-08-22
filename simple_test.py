"""
Simplified version for testing without heavy video/audio dependencies
"""
import asyncio
import os
from pathlib import Path
from typing import Dict, Any
import argparse
from loguru import logger
from clipfarming import ConfigManager
from clipfarming.agents import AgentOrchestrator, BaseAgent


class SimpleScraperAgent(BaseAgent):
    """Simplified scraper agent for testing"""
    
    def __init__(self, config):
        super().__init__("SimpleScraperAgent", config)
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        channel_url = input_data.get('channel_url', self.config.odysee.channel_url)
        self.logger.info(f"🔍 Scraping channel: {channel_url}")
        
        # Simulate discovering videos
        await asyncio.sleep(1)
        
        mock_videos = [
            {
                'url': f'{channel_url}/video1',
                'title': 'Amazing Content Video 1',
                'duration': 600,
                'filename': 'mock_video_1.mp4'
            },
            {
                'url': f'{channel_url}/video2', 
                'title': 'Interesting Tutorial Video 2',
                'duration': 480,
                'filename': 'mock_video_2.mp4'
            }
        ]
        
        self.logger.info(f"✅ Found {len(mock_videos)} videos")
        return {'downloaded_videos': mock_videos, 'channel_url': channel_url}


class SimpleClipperAgent(BaseAgent):
    """Simplified clipper agent for testing"""
    
    def __init__(self, config):
        super().__init__("SimpleClipperAgent", config)
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        videos = input_data.get('downloaded_videos', [])
        self.logger.info(f"✂️ Creating clips from {len(videos)} videos")
        
        all_clips = []
        for i, video in enumerate(videos):
            await asyncio.sleep(0.5)  # Simulate processing
            
            # Create mock clips
            for j in range(self.config.video.clips_per_video):
                clip = {
                    'filename': f'clip_{i}_{j}.mp4',
                    'title': f"{video['title']} - Clip {j+1}",
                    'duration': self.config.video.clip_length_seconds,
                    'source_video': video['title'],
                    'engagement_score': 0.8
                }
                all_clips.append(clip)
        
        self.logger.info(f"✅ Created {len(all_clips)} clips")
        return {'clips': all_clips, 'original_videos': videos}


class SimpleContentAgent(BaseAgent):
    """Simplified content generation agent for testing"""
    
    def __init__(self, config):
        super().__init__("SimpleContentAgent", config)
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        clips = input_data.get('clips', [])
        self.logger.info(f"📝 Generating content for {len(clips)} clips")
        
        enhanced_clips = []
        for clip in clips:
            await asyncio.sleep(0.2)  # Simulate AI processing
            
            enhanced_clip = clip.copy()
            enhanced_clip.update({
                'generated_title': f"🔥 {clip['title']} - VIRAL MOMENT!",
                'generated_description': f"Amazing highlight from {clip['source_video']}! You won't believe what happens next! 🤯",
                'generated_hashtags': ['#viral', '#trending', '#amazing', '#fyp'],
                'transcript': 'This is a simulated transcript of the video content...'
            })
            enhanced_clips.append(enhanced_clip)
        
        self.logger.info("✅ Generated content for all clips")
        return {'enhanced_clips': enhanced_clips, 'original_clips': clips}


class SimpleSocialAgent(BaseAgent):
    """Simplified social media agent for testing"""
    
    def __init__(self, config):
        super().__init__("SimpleSocialAgent", config)
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        clips = input_data.get('enhanced_clips', [])
        self.logger.info(f"📱 Posting {len(clips)} clips to social media")
        
        results = []
        for clip in clips:
            await asyncio.sleep(0.3)  # Simulate posting
            
            result = {
                'clip_title': clip['title'],
                'status': 'posted',
                'platforms': {
                    'tiktok': {'status': 'success', 'post_id': f"tiktok_{id(clip)}"},
                    'instagram': {'status': 'success', 'post_id': f"ig_{id(clip)}"},
                    'youtube': {'status': 'success', 'video_id': f"yt_{id(clip)}"}
                }
            }
            results.append(result)
        
        self.logger.info("✅ Posted all clips successfully")
        return {'posting_results': results, 'processed_clips': clips}


class SimpleClipfarmingSystem:
    """Simplified system for testing"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        self.orchestrator = AgentOrchestrator(self.config)
        self._setup_logging()
        self.config_manager.ensure_directories()
    
    def _setup_logging(self):
        """Setup simplified logging"""
        logger.remove()
        logger.add(
            lambda msg: print(msg, end=""),
            level=self.config.logging.level,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
        )
    
    async def setup(self):
        """Setup simplified agents"""
        logger.info("🚀 Setting up Simplified Clipfarming System")
        
        self.orchestrator.add_agent(SimpleScraperAgent(self.config))
        self.orchestrator.add_agent(SimpleClipperAgent(self.config))
        self.orchestrator.add_agent(SimpleContentAgent(self.config))
        self.orchestrator.add_agent(SimpleSocialAgent(self.config))
        
        await self.orchestrator.setup_all()
        logger.info("✅ System setup complete")
    
    async def run(self, channel_url: str = None) -> Dict[str, Any]:
        """Run simplified pipeline"""
        logger.info("🎬 Starting Clipfarming Pipeline")
        
        input_data = {
            'channel_url': channel_url or self.config.odysee.channel_url
        }
        
        result = await self.orchestrator.execute_pipeline(input_data)
        self._log_summary(result)
        
        await self.orchestrator.cleanup_all()
        logger.info("🎉 Pipeline completed successfully!")
        return result
    
    def _log_summary(self, result: Dict[str, Any]):
        """Log summary"""
        results = result.get('posting_results', [])
        logger.info("=== SUMMARY ===")
        logger.info(f"📊 Total clips: {len(results)}")
        logger.info(f"✅ All clips posted successfully!")
        logger.info("===============")


async def main():
    """Simple main function for testing"""
    parser = argparse.ArgumentParser(description="Simplified Clipfarming Test")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--channel", help="Channel URL")
    parser.add_argument("--setup-only", action="store_true", help="Setup only")
    
    args = parser.parse_args()
    
    system = SimpleClipfarmingSystem(args.config)
    await system.setup()
    
    if args.setup_only:
        print("✅ Setup completed successfully!")
        return
    
    channel_url = args.channel or "https://odysee.com/@example"
    await system.run(channel_url)


if __name__ == "__main__":
    asyncio.run(main())