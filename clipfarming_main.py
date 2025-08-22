"""
Clipfarming System Main Module
"""
import asyncio
import os
from pathlib import Path
from typing import Dict, Any
import argparse
from loguru import logger
from clipfarming import ConfigManager
from clipfarming.agents import AgentOrchestrator
from clipfarming.agents.scraper import OdyseeScraperAgent
from clipfarming.agents.clipper import VideoClippingAgent
from clipfarming.agents.content_generator import ContentGenerationAgent
from clipfarming.agents.social_media import SocialMediaAgent


class ClipfarmingSystem:
    """Main clipfarming system orchestrator"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        self.orchestrator = AgentOrchestrator(self.config)
        
        # Setup logging
        self._setup_logging()
        
        # Ensure directories exist
        self.config_manager.ensure_directories()
    
    def _setup_logging(self):
        """Configure logging based on configuration"""
        logger.remove()  # Remove default handler
        
        # Add console handler if enabled
        if self.config.logging.console_output:
            logger.add(
                lambda msg: print(msg, end=""),
                level=self.config.logging.level,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{extra[agent]}</cyan> | <level>{message}</level>"
            )
        
        # Add file handler
        log_file_path = Path(self.config.logging.file_path)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            str(log_file_path),
            level=self.config.logging.level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[agent]} | {message}",
            rotation="1 week",
            retention="1 month"
        )
        
        logger.bind(agent="ClipfarmingSystem")
    
    async def setup(self):
        """Initialize the clipfarming system"""
        logger.info("Setting up Clipfarming System")
        
        # Create and add agents
        scraper_agent = OdyseeScraperAgent(self.config)
        clipper_agent = VideoClippingAgent(self.config)
        content_agent = ContentGenerationAgent(self.config)
        social_agent = SocialMediaAgent(self.config)
        
        self.orchestrator.add_agent(scraper_agent)
        self.orchestrator.add_agent(clipper_agent)
        self.orchestrator.add_agent(content_agent)
        self.orchestrator.add_agent(social_agent)
        
        # Setup all agents
        await self.orchestrator.setup_all()
        
        logger.info("Clipfarming System setup complete")
    
    async def run(self, channel_url: str = None) -> Dict[str, Any]:
        """
        Run the complete clipfarming pipeline
        
        Args:
            channel_url: Optional channel URL to override config
            
        Returns:
            Pipeline execution results
        """
        try:
            logger.info("Starting Clipfarming Pipeline")
            
            # Prepare input data
            input_data = {
                'channel_url': channel_url or self.config.odysee.channel_url
            }
            
            # Execute the pipeline
            result = await self.orchestrator.execute_pipeline(input_data)
            
            # Log summary
            self._log_summary(result)
            
            logger.info("Clipfarming Pipeline completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Clipfarming Pipeline failed: {e}")
            raise
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Cleanup system resources"""
        logger.info("Cleaning up Clipfarming System")
        await self.orchestrator.cleanup_all()
        
        # Clean temp files if configured
        if self.config.storage.clean_temp_after_processing:
            temp_dir = Path(self.config.storage.temp_dir)
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned temp directory: {temp_dir}")
    
    def _log_summary(self, result: Dict[str, Any]):
        """Log a summary of the pipeline results"""
        try:
            posting_results = result.get('posting_results', [])
            
            logger.info("=== CLIPFARMING SUMMARY ===")
            logger.info(f"Total clips processed: {len(posting_results)}")
            
            # Count successful posts by platform
            platform_stats = {}
            for clip_result in posting_results:
                platforms = clip_result.get('platforms', {})
                for platform, post_result in platforms.items():
                    if platform not in platform_stats:
                        platform_stats[platform] = {'success': 0, 'error': 0}
                    
                    if post_result.get('status') == 'success':
                        platform_stats[platform]['success'] += 1
                    else:
                        platform_stats[platform]['error'] += 1
            
            # Log platform statistics
            for platform, stats in platform_stats.items():
                total = stats['success'] + stats['error']
                success_rate = (stats['success'] / total * 100) if total > 0 else 0
                logger.info(f"{platform.upper()}: {stats['success']}/{total} successful ({success_rate:.1f}%)")
            
            logger.info("=== END SUMMARY ===")
            
        except Exception as e:
            logger.warning(f"Error generating summary: {e}")


async def main():
    """Main entry point for the clipfarming system"""
    parser = argparse.ArgumentParser(description="Automated Clipfarming System")
    parser.add_argument(
        "--config", 
        default="config.yaml", 
        help="Path to configuration file"
    )
    parser.add_argument(
        "--channel", 
        help="Odysee channel URL to process"
    )
    parser.add_argument(
        "--setup-only", 
        action="store_true", 
        help="Only setup the system without running pipeline"
    )
    
    args = parser.parse_args()
    
    # Create and setup system
    system = ClipfarmingSystem(args.config)
    await system.setup()
    
    if args.setup_only:
        print("System setup completed. Use --channel URL to run the pipeline.")
        return
    
    # Validate requirements
    if not args.channel:
        config = system.config
        if not config.odysee.channel_url or config.odysee.channel_url == "https://odysee.com/@channelname":
            print("Error: No channel URL provided. Use --channel URL or update config.yaml")
            return
    
    # Check for required API keys
    if not os.getenv('OPENAI_API_KEY'):
        print("Warning: OPENAI_API_KEY environment variable not set. AI features may not work.")
    
    try:
        # Run the pipeline
        await system.run(args.channel)
        print("✅ Clipfarming pipeline completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())