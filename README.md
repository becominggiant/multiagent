# Automated Clipfarming System

A Python-based automated clipfarming program that uses a team of AI agents to extract videos from Odysee channels, create engaging clips, and automatically post them to multiple social media platforms.

## Features

### 🎯 Core Functionality
- **Video Extraction**: Scrape and download videos from Odysee channels
- **Intelligent Clipping**: AI-powered segment detection based on audio engagement analysis  
- **Content Enhancement**: Automatic subtitle generation, dead air removal, branding overlays
- **AI Content Generation**: GPT-powered titles, descriptions, and hashtags
- **Multi-Platform Posting**: Automated posting to TikTok, Instagram Reels, YouTube Shorts, Twitter

### 🤖 Agent-Based Architecture
- **Scraper Agent**: Downloads videos from Odysee channels
- **Clipper Agent**: Creates and edits video clips using AI analysis
- **Content Agent**: Generates titles, descriptions, and transcripts using Whisper & GPT
- **Social Media Agent**: Posts clips to multiple platforms with optimal scheduling

### 🔧 AI Integrations
- **OpenAI Whisper**: Speech-to-text transcription
- **OpenAI GPT**: Content generation for titles and descriptions
- **Audio Analysis**: Librosa for engagement-based segment detection
- **Computer Vision**: Optional scene analysis capabilities

## Quick Start

### Prerequisites
- Python 3.9+
- OpenAI API key
- Social media platform API keys (optional for full functionality)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd multiagent
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

4. **Configure the system**
Edit `config.yaml` to customize:
- Video processing parameters
- AI model settings  
- Social media platforms
- Clip generation settings

### Basic Usage

**Run with a specific Odysee channel:**
```bash
python main.py --channel "https://odysee.com/@channelname"
```

**Setup only (test configuration):**
```bash
python main.py --setup-only
```

**Use custom config file:**
```bash
python main.py --config my-config.yaml --channel "https://odysee.com/@example"
```

## Configuration

The system uses a YAML configuration file (`config.yaml`) to control all parameters:

### Key Configuration Sections
- **`odysee`**: Channel URL, video limits, age filters
- **`video`**: Clip length, clips per video, output settings
- **`ai`**: Model selection, prompts, API settings
- **`editing`**: Subtitle, branding, transition settings
- **`platforms`**: Enable/disable platforms, posting schedules, hashtags
- **`performance`**: Concurrency limits, GPU acceleration
- **`storage`**: Directory paths, cleanup settings

## Architecture

### Agent Pipeline
```
Odysee Channel URL
    ↓
[Scraper Agent] → Downloads videos
    ↓
[Clipper Agent] → Creates clips with AI analysis
    ↓
[Content Agent] → Generates titles, descriptions, transcripts
    ↓
[Social Media Agent] → Posts to platforms
    ↓
Results & Analytics
```

### Key Components
- **ConfigManager**: Handles YAML configuration with validation
- **AgentOrchestrator**: Manages agent lifecycle and pipeline execution
- **BaseAgent**: Abstract base class for all agents
- **Platform Integrations**: Modular social media posting capabilities

## AI Features

### Engagement-Based Clipping
- Audio energy analysis using Librosa
- Spectral centroid analysis for content brightness
- Zero-crossing rate for speech activity detection
- Combined engagement scoring for optimal segment selection

### Content Generation
- **Whisper Integration**: Automatic speech transcription
- **GPT-4 Content**: Engaging titles and descriptions
- **Smart Hashtags**: AI-generated relevant hashtags
- **Subtitle Generation**: SRT format subtitle files

## Platform Support

### Currently Integrated
- **TikTok**: Short-form video posting (simulated)
- **Instagram Reels**: Vertical video content
- **YouTube Shorts**: Short-form YouTube content  
- **Twitter/X**: Video posts with text

### API Requirements
Each platform requires appropriate API keys and developer access. The system includes simulation modes for testing without full API access.

## Advanced Usage

### Custom Agent Development
Extend the system by creating custom agents:

```python
from clipfarming.agents import BaseAgent

class MyCustomAgent(BaseAgent):
    async def execute(self, input_data):
        # Your custom logic here
        return processed_data
```

### Performance Tuning
- Adjust concurrent processing limits in `config.yaml`
- Enable GPU acceleration for video processing
- Configure caching and temp file management

### Monitoring and Logging
- Structured logging with Loguru
- Performance metrics tracking
- Error handling and recovery

## Development

### Project Structure
```
multiagent/
├── clipfarming/
│   ├── __init__.py              # Config management
│   └── agents/
│       ├── __init__.py          # Base agent classes
│       ├── scraper.py           # Odysee video scraping
│       ├── clipper.py           # Video clipping and editing
│       ├── content_generator.py # AI content generation
│       └── social_media.py      # Social media posting
├── config.yaml                 # System configuration
├── requirements.txt            # Python dependencies  
├── main.py                     # Main entry point
└── clipfarming_main.py        # Core system implementation
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Implement your changes with tests
4. Submit a pull request

## License

[Add your license information here]

## Support

For issues and questions:
1. Check the configuration documentation
2. Review logs in `./logs/clipfarming.log`
3. Open an issue on GitHub with error details

---

**⚠️ Important**: This system is for educational and research purposes. Ensure compliance with platform terms of service and content policies when using for production.