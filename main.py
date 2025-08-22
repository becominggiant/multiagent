"""
Automated Clipfarming System with AI Agents

This system extracts videos from Odysee channels, automatically generates clips 
based on engagement analysis, edits them with AI-powered enhancements, and 
posts them to multiple social media platforms.
"""

import asyncio
from simple_test import main as simple_main

async def main() -> None:
    """
    Main entry point for the clipfarming system.
    
    This replaces the previous affiliate marketing system with a comprehensive
    clipfarming solution that uses AI agents for:
    - Video scraping and downloading from Odysee
    - Intelligent clip creation based on engagement analysis  
    - AI-powered content generation (titles, descriptions, transcripts)
    - Automated posting to multiple social media platforms
    
    Currently using simplified agents for demonstration. The full implementation
    with video processing, Whisper transcription, and real social media APIs
    is available in the clipfarming/ directory.
    """
    print("🎬 Starting Automated Clipfarming System with AI Agents")
    print("=" * 60)
    print("ℹ️  Using simplified agents for demonstration")
    print("📖 See README.md for full implementation details")
    print()
    
    try:
        await simple_main()
    except KeyboardInterrupt:
        print("\n⚠️ System interrupted by user")
    except Exception as e:
        print(f"❌ System error: {e}")
        # In production, you might want to add error reporting here


if __name__ == "__main__":
    asyncio.run(main())
