"""
Odysee Video Scraper and Downloader Agent
"""
import asyncio
import aiohttp
from pathlib import Path
from typing import List, Dict, Any, Optional
import yt_dlp
from bs4 import BeautifulSoup
import re
from clipfarming.agents import BaseAgent


class OdyseeScraperAgent(BaseAgent):
    """Agent responsible for scraping and downloading videos from Odysee channels"""
    
    def __init__(self, config):
        super().__init__("OdyseeScraperAgent", config)
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def setup(self):
        """Initialize the HTTP session"""
        await super().setup()
        self.session = aiohttp.ClientSession()
    
    async def cleanup(self):
        """Cleanup the HTTP session"""
        if self.session:
            await self.session.close()
        await super().cleanup()
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method - scrapes channel and downloads videos
        
        Args:
            input_data: Dictionary containing channel_url
            
        Returns:
            Dictionary containing downloaded video information
        """
        channel_url = input_data.get('channel_url', self.config.odysee.channel_url)
        
        self.logger.info(f"Starting to scrape Odysee channel: {channel_url}")
        
        # Get video URLs from the channel
        video_urls = await self._get_channel_videos(channel_url)
        
        # Limit the number of videos based on config
        video_urls = video_urls[:self.config.odysee.max_videos]
        
        # Download videos concurrently
        downloaded_videos = await self._download_videos(video_urls)
        
        return {
            'downloaded_videos': downloaded_videos,
            'channel_url': channel_url
        }
    
    async def _get_channel_videos(self, channel_url: str) -> List[str]:
        """
        Scrape video URLs from an Odysee channel
        
        Args:
            channel_url: The Odysee channel URL
            
        Returns:
            List of video URLs
        """
        try:
            self.logger.info(f"Fetching channel page: {channel_url}")
            
            async with self.session.get(channel_url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to fetch channel page: HTTP {response.status}")
                
                html_content = await response.text()
            
            # Parse HTML to find video URLs
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find video links - Odysee typically uses specific patterns
            video_links = []
            
            # Look for links that match Odysee video URL patterns
            for link in soup.find_all('a', href=True):
                href = link['href']
                if self._is_video_url(href):
                    if href.startswith('/'):
                        href = f"https://odysee.com{href}"
                    video_links.append(href)
            
            # Remove duplicates
            video_links = list(set(video_links))
            
            self.logger.info(f"Found {len(video_links)} video URLs")
            return video_links
            
        except Exception as e:
            self.logger.error(f"Error scraping channel: {e}")
            # Fallback: try using yt-dlp to get playlist
            return await self._get_videos_with_ytdlp(channel_url)
    
    def _is_video_url(self, url: str) -> bool:
        """Check if URL is a video URL based on Odysee patterns"""
        # Odysee video URLs typically contain specific patterns
        video_patterns = [
            r'/[^/]+/[^/]+$',  # Basic video pattern
            r'/@[^/]+/[^/]+',   # Channel/video pattern
        ]
        
        for pattern in video_patterns:
            if re.search(pattern, url):
                return True
        return False
    
    async def _get_videos_with_ytdlp(self, channel_url: str) -> List[str]:
        """
        Fallback method using yt-dlp to get video URLs
        
        Args:
            channel_url: The channel URL
            
        Returns:
            List of video URLs
        """
        try:
            self.logger.info("Using yt-dlp as fallback for video discovery")
            
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True,
                'playlistend': self.config.odysee.max_videos
            }
            
            # Run yt-dlp in a thread to avoid blocking
            def get_info():
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(channel_url, download=False)
                    if 'entries' in info:
                        return [entry.get('url', entry.get('webpage_url', '')) 
                               for entry in info['entries'] if entry]
                    return []
            
            loop = asyncio.get_event_loop()
            video_urls = await loop.run_in_executor(None, get_info)
            
            self.logger.info(f"Found {len(video_urls)} videos using yt-dlp")
            return video_urls
            
        except Exception as e:
            self.logger.error(f"Error with yt-dlp fallback: {e}")
            return []
    
    async def _download_videos(self, video_urls: List[str]) -> List[Dict[str, Any]]:
        """
        Download videos concurrently
        
        Args:
            video_urls: List of video URLs to download
            
        Returns:
            List of downloaded video information
        """
        semaphore = asyncio.Semaphore(self.config.performance.max_concurrent_downloads)
        
        async def download_single(url: str) -> Optional[Dict[str, Any]]:
            async with semaphore:
                return await self._download_video(url)
        
        # Create download tasks
        tasks = [download_single(url) for url in video_urls]
        
        # Wait for all downloads to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful downloads
        downloaded_videos = []
        for result in results:
            if isinstance(result, dict):
                downloaded_videos.append(result)
            elif isinstance(result, Exception):
                self.logger.error(f"Download error: {result}")
        
        return downloaded_videos
    
    async def _download_video(self, video_url: str) -> Optional[Dict[str, Any]]:
        """
        Download a single video
        
        Args:
            video_url: The video URL to download
            
        Returns:
            Dictionary with video information or None if failed
        """
        try:
            self.logger.info(f"Downloading video: {video_url}")
            
            # Create output directory
            output_dir = Path(self.config.storage.temp_dir) / "videos"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # yt-dlp options
            ydl_opts = {
                'outtmpl': str(output_dir / '%(title)s.%(ext)s'),
                'format': 'best[height<=1080]',  # Limit to 1080p or lower
                'writeinfojson': True,  # Save metadata
            }
            
            # Download in executor to avoid blocking
            def download():
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(video_url, download=True)
                    return {
                        'url': video_url,
                        'title': info.get('title', ''),
                        'duration': info.get('duration', 0),
                        'upload_date': info.get('upload_date', ''),
                        'filename': ydl.prepare_filename(info),
                        'thumbnail': info.get('thumbnail', ''),
                        'description': info.get('description', ''),
                        'view_count': info.get('view_count', 0),
                        'like_count': info.get('like_count', 0)
                    }
            
            loop = asyncio.get_event_loop()
            video_info = await loop.run_in_executor(None, download)
            
            self.logger.info(f"Successfully downloaded: {video_info['title']}")
            return video_info
            
        except Exception as e:
            self.logger.error(f"Failed to download video {video_url}: {e}")
            return None