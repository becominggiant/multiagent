"""
Social Media Platform Posting Agent
"""
import asyncio
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
from clipfarming.agents import BaseAgent


class SocialMediaAgent(BaseAgent):
    """Agent responsible for posting clips to various social media platforms"""
    
    def __init__(self, config):
        super().__init__("SocialMediaAgent", config)
        self.platform_clients = {}
    
    async def setup(self):
        """Initialize social media platform clients"""
        await super().setup()
        
        # Initialize platform clients based on configuration
        if self.config.platforms.tiktok.enabled:
            await self._setup_tiktok()
        
        if self.config.platforms.instagram.enabled:
            await self._setup_instagram()
        
        if self.config.platforms.youtube_shorts.enabled:
            await self._setup_youtube()
        
        if self.config.platforms.twitter.enabled:
            await self._setup_twitter()
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method - posts clips to social media platforms
        
        Args:
            input_data: Dictionary containing enhanced clips
            
        Returns:
            Dictionary containing posting results
        """
        enhanced_clips = input_data.get('enhanced_clips', [])
        
        self.logger.info(f"Posting {len(enhanced_clips)} clips to social media platforms")
        
        posting_results = []
        
        # Process clips with rate limiting
        semaphore = asyncio.Semaphore(self.config.performance.max_concurrent_uploads)
        
        async def post_clip(clip_info):
            async with semaphore:
                return await self._post_clip_to_platforms(clip_info)
        
        tasks = [post_clip(clip) for clip in enhanced_clips]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, dict):
                posting_results.append(result)
            elif isinstance(result, Exception):
                self.logger.error(f"Error posting clip: {result}")
        
        return {
            'posting_results': posting_results,
            'processed_clips': enhanced_clips
        }
    
    async def _post_clip_to_platforms(self, clip_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post a single clip to all enabled platforms
        
        Args:
            clip_info: Enhanced clip information
            
        Returns:
            Posting results for the clip
        """
        try:
            self.logger.info(f"Posting clip to platforms: {clip_info['title']}")
            
            clip_results = {
                'clip_title': clip_info['title'],
                'clip_file': clip_info['filename'],
                'platforms': {}
            }
            
            # Determine posting schedule
            should_post_now = await self._should_post_now(clip_info)
            
            if not should_post_now:
                self.logger.info(f"Scheduling clip for later: {clip_info['title']}")
                await self._schedule_for_later(clip_info)
                clip_results['status'] = 'scheduled'
                return clip_results
            
            # Post to each enabled platform
            platform_tasks = []
            
            if self.config.platforms.tiktok.enabled:
                platform_tasks.append(self._post_to_tiktok(clip_info))
            
            if self.config.platforms.instagram.enabled:
                platform_tasks.append(self._post_to_instagram(clip_info))
            
            if self.config.platforms.youtube_shorts.enabled:
                platform_tasks.append(self._post_to_youtube(clip_info))
            
            if self.config.platforms.twitter.enabled:
                platform_tasks.append(self._post_to_twitter(clip_info))
            
            # Execute all platform posts concurrently
            platform_results = await asyncio.gather(*platform_tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(platform_results):
                platform_name = ['tiktok', 'instagram', 'youtube_shorts', 'twitter'][i]
                if isinstance(result, dict):
                    clip_results['platforms'][platform_name] = result
                else:
                    clip_results['platforms'][platform_name] = {
                        'status': 'error',
                        'error': str(result)
                    }
            
            clip_results['status'] = 'posted'
            return clip_results
            
        except Exception as e:
            self.logger.error(f"Error posting clip {clip_info['title']}: {e}")
            return {
                'clip_title': clip_info['title'],
                'status': 'error',
                'error': str(e)
            }
    
    async def _should_post_now(self, clip_info: Dict[str, Any]) -> bool:
        """
        Determine if clip should be posted immediately or scheduled
        
        Args:
            clip_info: Clip information
            
        Returns:
            True if should post now, False if should schedule
        """
        # For now, implement simple immediate posting
        # In a full implementation, this would check optimal posting times
        return True
    
    async def _schedule_for_later(self, clip_info: Dict[str, Any]):
        """
        Schedule clip for later posting
        
        Args:
            clip_info: Clip information
        """
        # Store clip info for later posting
        schedule_dir = Path(self.config.storage.cache_dir) / "scheduled_posts"
        schedule_dir.mkdir(parents=True, exist_ok=True)
        
        schedule_file = schedule_dir / f"scheduled_{datetime.now().isoformat()}.json"
        
        with open(schedule_file, 'w') as f:
            json.dump(clip_info, f, indent=2, default=str)
        
        self.logger.info(f"Scheduled clip for later posting: {schedule_file}")
    
    async def _setup_tiktok(self):
        """Setup TikTok API client"""
        # In a real implementation, this would setup TikTok API client
        # For now, we'll simulate the setup
        self.logger.info("TikTok client setup (simulated)")
        self.platform_clients['tiktok'] = "tiktok_client_placeholder"
    
    async def _setup_instagram(self):
        """Setup Instagram API client"""
        # In a real implementation, this would setup Instagram Graph API or instagrapi
        self.logger.info("Instagram client setup (simulated)")
        self.platform_clients['instagram'] = "instagram_client_placeholder"
    
    async def _setup_youtube(self):
        """Setup YouTube API client"""
        # In a real implementation, this would setup YouTube Data API client
        self.logger.info("YouTube client setup (simulated)")
        self.platform_clients['youtube'] = "youtube_client_placeholder"
    
    async def _setup_twitter(self):
        """Setup Twitter/X API client"""
        # In a real implementation, this would setup Twitter API v2 client
        self.logger.info("Twitter client setup (simulated)")
        self.platform_clients['twitter'] = "twitter_client_placeholder"
    
    async def _post_to_tiktok(self, clip_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post clip to TikTok
        
        Args:
            clip_info: Clip information
            
        Returns:
            Posting result
        """
        try:
            self.logger.info(f"Posting to TikTok: {clip_info['title']}")
            
            # In a real implementation, this would use TikTok API
            # For demonstration, we'll simulate the post
            
            # Prepare content
            title = clip_info.get('generated_title', clip_info['title'])[:100]  # TikTok limit
            hashtags = self.config.platforms.tiktok.hashtags.copy()
            
            # Add generated hashtags
            if 'generated_hashtags' in clip_info:
                hashtags.extend(clip_info['generated_hashtags'][:5])  # Limit hashtags
            
            caption = f"{title} {' '.join(hashtags)}"
            
            # Simulate posting delay
            await asyncio.sleep(2)
            
            # Simulate successful post
            post_id = f"tiktok_{datetime.now().timestamp()}"
            
            self.logger.info(f"Successfully posted to TikTok: {post_id}")
            
            return {
                'status': 'success',
                'post_id': post_id,
                'platform': 'tiktok',
                'url': f"https://tiktok.com/@user/video/{post_id}",
                'caption': caption
            }
            
        except Exception as e:
            self.logger.error(f"Failed to post to TikTok: {e}")
            return {
                'status': 'error',
                'platform': 'tiktok',
                'error': str(e)
            }
    
    async def _post_to_instagram(self, clip_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post clip to Instagram Reels
        
        Args:
            clip_info: Clip information
            
        Returns:
            Posting result
        """
        try:
            self.logger.info(f"Posting to Instagram: {clip_info['title']}")
            
            # Prepare content
            title = clip_info.get('generated_title', clip_info['title'])
            description = clip_info.get('generated_description', '')
            hashtags = self.config.platforms.instagram.hashtags.copy()
            
            # Add generated hashtags
            if 'generated_hashtags' in clip_info:
                hashtags.extend(clip_info['generated_hashtags'][:10])
            
            caption = f"{title}\n\n{description}\n\n{' '.join(hashtags)}"
            
            # Simulate posting delay
            await asyncio.sleep(3)
            
            # Simulate successful post
            post_id = f"instagram_{datetime.now().timestamp()}"
            
            self.logger.info(f"Successfully posted to Instagram: {post_id}")
            
            return {
                'status': 'success',
                'post_id': post_id,
                'platform': 'instagram',
                'url': f"https://instagram.com/p/{post_id}",
                'caption': caption[:2200]  # Instagram limit
            }
            
        except Exception as e:
            self.logger.error(f"Failed to post to Instagram: {e}")
            return {
                'status': 'error',
                'platform': 'instagram',
                'error': str(e)
            }
    
    async def _post_to_youtube(self, clip_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post clip to YouTube Shorts
        
        Args:
            clip_info: Clip information
            
        Returns:
            Posting result
        """
        try:
            self.logger.info(f"Posting to YouTube Shorts: {clip_info['title']}")
            
            # Prepare content
            title = clip_info.get('generated_title', clip_info['title'])[:100]  # YouTube limit
            description = clip_info.get('generated_description', '')
            hashtags = self.config.platforms.youtube_shorts.hashtags.copy()
            
            # Add generated hashtags
            if 'generated_hashtags' in clip_info:
                hashtags.extend(clip_info['generated_hashtags'][:15])
            
            full_description = f"{description}\n\n{' '.join(hashtags)}"
            
            # Simulate posting delay
            await asyncio.sleep(5)
            
            # Simulate successful post
            video_id = f"youtube_{datetime.now().timestamp()}"
            
            self.logger.info(f"Successfully posted to YouTube: {video_id}")
            
            return {
                'status': 'success',
                'video_id': video_id,
                'platform': 'youtube_shorts',
                'url': f"https://youtube.com/shorts/{video_id}",
                'title': title,
                'description': full_description[:5000]  # YouTube limit
            }
            
        except Exception as e:
            self.logger.error(f"Failed to post to YouTube: {e}")
            return {
                'status': 'error',
                'platform': 'youtube_shorts',
                'error': str(e)
            }
    
    async def _post_to_twitter(self, clip_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post clip to Twitter/X
        
        Args:
            clip_info: Clip information
            
        Returns:
            Posting result
        """
        try:
            self.logger.info(f"Posting to Twitter: {clip_info['title']}")
            
            # Prepare content
            title = clip_info.get('generated_title', clip_info['title'])
            hashtags = self.config.platforms.twitter.hashtags.copy()
            
            # Add generated hashtags
            if 'generated_hashtags' in clip_info:
                hashtags.extend(clip_info['generated_hashtags'][:5])
            
            # Twitter has character limits
            tweet_text = f"{title} {' '.join(hashtags)}"
            if len(tweet_text) > 280:
                tweet_text = tweet_text[:277] + "..."
            
            # Simulate posting delay
            await asyncio.sleep(2)
            
            # Simulate successful post
            tweet_id = f"twitter_{datetime.now().timestamp()}"
            
            self.logger.info(f"Successfully posted to Twitter: {tweet_id}")
            
            return {
                'status': 'success',
                'tweet_id': tweet_id,
                'platform': 'twitter',
                'url': f"https://twitter.com/user/status/{tweet_id}",
                'text': tweet_text
            }
            
        except Exception as e:
            self.logger.error(f"Failed to post to Twitter: {e}")
            return {
                'status': 'error',
                'platform': 'twitter',
                'error': str(e)
            }