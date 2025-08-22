"""
Clipfarming System Configuration Manager
"""
import yaml
import os
from typing import Dict, Any
from pydantic import BaseModel, Field
from pathlib import Path


class OdyseeConfig(BaseModel):
    channel_url: str = "https://odysee.com/@channelname"
    max_videos: int = 10
    video_age_limit_days: int = 30


class VideoConfig(BaseModel):
    clip_length_seconds: int = 30
    clips_per_video: int = 3
    min_engagement_threshold: float = 0.1
    output_format: str = "mp4"
    output_resolution: str = "1080p"


class AIConfig(BaseModel):
    openai_model: str = "gpt-4"
    whisper_model: str = "base"
    vision_model: str = "gpt-4-vision-preview"
    title_prompt: str = "Generate an engaging title for this video clip about: {topic}"
    description_prompt: str = "Write a compelling description for this {duration}s video clip"


class EditingConfig(BaseModel):
    add_subtitles: bool = True
    remove_dead_air: bool = True
    dead_air_threshold: float = 2.0
    add_transitions: bool = True
    add_branding: bool = True
    branding_logo_path: str = "assets/logo.png"
    branding_position: str = "bottom-right"


class PlatformConfig(BaseModel):
    enabled: bool = True
    post_schedule: str = "prime_time"
    hashtags: list[str] = Field(default_factory=list)


class PlatformsConfig(BaseModel):
    tiktok: PlatformConfig = PlatformConfig(hashtags=["#viral", "#trending", "#fyp"])
    instagram: PlatformConfig = PlatformConfig(hashtags=["#reels", "#viral", "#trending"])
    youtube_shorts: PlatformConfig = PlatformConfig(hashtags=["#shorts", "#viral", "#trending"])
    twitter: PlatformConfig = PlatformConfig(enabled=False, hashtags=["#video", "#content"])


class PerformanceConfig(BaseModel):
    max_concurrent_downloads: int = 3
    max_concurrent_clips: int = 2
    max_concurrent_uploads: int = 1
    enable_gpu_acceleration: bool = True


class StorageConfig(BaseModel):
    temp_dir: str = "./temp"
    output_dir: str = "./output"
    cache_dir: str = "./cache"
    clean_temp_after_processing: bool = True


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file_path: str = "./logs/clipfarming.log"
    console_output: bool = True


class ClipfarmingConfig(BaseModel):
    odysee: OdyseeConfig = OdyseeConfig()
    video: VideoConfig = VideoConfig()
    ai: AIConfig = AIConfig()
    editing: EditingConfig = EditingConfig()
    platforms: PlatformsConfig = PlatformsConfig()
    performance: PerformanceConfig = PerformanceConfig()
    storage: StorageConfig = StorageConfig()
    logging: LoggingConfig = LoggingConfig()


class ConfigManager:
    """Manages configuration loading and validation"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config: ClipfarmingConfig = self._load_config()
    
    def _load_config(self) -> ClipfarmingConfig:
        """Load and validate configuration from YAML file"""
        if not self.config_path.exists():
            # Create default config if it doesn't exist
            default_config = ClipfarmingConfig()
            self._save_config(default_config)
            return default_config
        
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            return ClipfarmingConfig(**config_data)
        except Exception as e:
            print(f"Error loading config: {e}")
            return ClipfarmingConfig()
    
    def _save_config(self, config: ClipfarmingConfig):
        """Save configuration to YAML file"""
        with open(self.config_path, 'w') as f:
            yaml.dump(config.model_dump(), f, default_flow_style=False, indent=2)
    
    def get_config(self) -> ClipfarmingConfig:
        """Get the current configuration"""
        return self.config
    
    def reload_config(self):
        """Reload configuration from file"""
        self.config = self._load_config()
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        dirs_to_create = [
            self.config.storage.temp_dir,
            self.config.storage.output_dir,
            self.config.storage.cache_dir,
            Path(self.config.logging.file_path).parent,
            "assets"
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)