<h1 align="center">
  <br>
    <img src="https://i.imgur.com/i0TtYWh.png" alt="Voice Oasis Logo" width="200">
    <br>
    Voice Oasis
    <br>
</h1>

<h4 align="center">An intelligent voice enhancement and memory platform for conversational agents</h4>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#technologies">Technologies</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#privacy-dashboard">Privacy Dashboard</a> •
  <a href="#system-requirements">System Requirements</a> •
  <a href="#project-structure">Project Structure</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#contributors">Contributors</a> •
  <a href="#license">License</a>
</p>

## Key Features

🎙 *Intelligent Voice Processing*
* Isolation of the main voice from background noise and other voices
* AI-powered voice recognition and speaker identification
* High-quality speech-to-text (STT) with focus on the main speaker

🧠 *Memory Management*
* Extraction and storage of conversation information as "memories" per voice
* Context-based integration of memories for more emotional and informative conversations
* Sentiment analysis and conversation personalization

🔒 *Privacy & Security*
* User-friendly privacy dashboard for complete data transparency
* Easy way for users to view and delete their stored voice data
* Strict privacy policies and secure data storage

🤖 *API Integration*
* Seamless interaction with the Helbling VoiceBot Hemy
* Extensible API for integration with other conversational agents
* Real-time audio preprocessing with low latency

## Technologies

The project uses numerous modern technologies:

* *Python 3.12*: As the main programming language
* *SpeechBrain*: For speaker recognition and identification
* *Azure Cognitive Services*: For high-quality speech recognition
* *OpenAI*: For advanced text analysis and generation
* *Flask*: For the backend and API endpoints
* *WebSockets*: For real-time audio data streaming
* *PyTorch & Torchaudio*: For advanced audio processing
* *Noisereduce & Librosa*: For audio enhancement and noise reduction

## Installation

### Prerequisites
- Python 3.12
- pip and pipx

### Local Installation

bash
# Install poetry (Python package manager)
$ pipx install poetry

# Install dependencies
$ poetry install

# Start virtual environment
$ poetry shell

# Start the application
$ python ./src/relay.py


### Environment Variables

Create a .env file in the main directory with your API keys:


SPEECH_KEY=your_azure_speech_key
SPEECH_REGION=your_azure_region
OPENAI_API_KEY=your_openai_api_key


## Usage

### Relay Server

The relay server provides an API for processing audio inputs and interacting with the VoiceBot:

bash
# Start the relay server
$ python ./src/relay.py

# Start with privacy dashboard enabled
$ python ./src/connect_privacy_dashboard.py


After starting, the API is available at http://localhost:5000 and the Swagger documentation at http://localhost:5000/apidocs/.

### Web App

The web app can be started at [voiceoasis.azurewebsites.net](https://voiceoasis.azurewebsites.net/).

Optional URL parameters:
* relayURL=ip:port: IP:port of the relay endpoint (default: localhost:5000)
* audioChunkLengthMS=number: Chunk length in milliseconds (default: 500)
* continuousListening=1: Enable continuous recording
* lang=language: Language (en-US, de-DE, fr-FR, it-IT, es-ES, ja-JP)

## Privacy Dashboard

The privacy dashboard allows users to view and delete their stored voice data:

bash
# Start the relay server with privacy dashboard enabled
$ python ./src/connect_privacy_dashboard.py


The dashboard is accessible at http://localhost:5000/privacy.

### Dashboard Features

- Display of all stored voice data and memories
- Overview of chat sessions linked to your voice profile
- Simple deletion of all data with one click
- Complete transparency about stored information

## System Requirements

* *Recommended Hardware*: Modern CPU with 4+ cores, at least 8GB RAM
* *Latency*: Processing of audio samples in less than 0.5 seconds on consumer-grade hardware
* *Storage Space*: At least 500MB free space for models and database

## Project Structure


voice_oasis/
├── src/                     # Source code
│   ├── relay.py             # Main server and API endpoints
│   ├── privacy_dashboard.py # Privacy dashboard
│   ├── audiosame.py         # Voice processing algorithms
│   ├── openai_api_example.py # Example for OpenAI integration
│   └── connect_privacy_dashboard.py # Connection to dashboard
├── voice_database/          # Stored voice data and memories
├── pretrained_models/       # Pretrained ML models
├── docs/                    # Documentation
├── pyproject.toml           # Project dependencies
└── poetry.lock              # Locked dependency versions


## Documentation

Comprehensive API documentation is available in Swagger. Start the relay server and visit:
- [localhost:5000/apidocs/](http://localhost:5000/apidocs/)

To generate the Swagger HTML documentation:
bash
npm install -g redoc-cli
redoc-cli bundle docs/apispec.json -o docs/apispec.html


## Contributors

This project was developed as part of the Helbling Hackathon.

## License

[MIT](LICENSE)
