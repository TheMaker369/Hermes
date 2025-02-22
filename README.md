# Hermes AI System

A distributed AI system based on the Kabbalistic Tree of Life architecture, integrating MAGI (Multi-Agent General Intelligence) and APRE (Automated Pattern Recognition Engine).

## Architecture

The system is structured according to the Tree of Life's Sephirot:

- **Chokmah (Wisdom)**: MAGI implementation for distributed reasoning
- **Binah (Understanding)**: APRE implementation for pattern recognition
- **Tiferet (Beauty)**: Response harmonization
- **Yesod (Foundation)**: Data persistence with ChromaDB
- **Gevurah (Severity)**: Security and validation
- **Netzach (Victory)**: External API integration

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env`:
```env
FIRECRAWL_API_KEY=your_key
ALLOW_EXTERNAL=true
ALLOW_REMOTE=true
LOCAL_ONLY_LLM=true
CHROMA_REMOTE=false
```

3. Run with Docker:
```bash
docker-compose up --build
```

## Testing

Run tests with pytest:
```bash
pytest tests/ -v --cov=hermes
```

## Components

### MAGI (Multi-Agent General Intelligence)
- Distributed reasoning system
- Three primary agents: Melchior, Balthasar, Casper
- Specializes in knowledge synthesis

### APRE (Automated Pattern Recognition Engine)
- Pattern detection and analysis
- Temporal and semantic pattern recognition
- ChromaDB integration for pattern storage

### External Integrations
- Firecrawl API integration
- Adaptive retry logic
- Circuit breaker pattern

## Security

- Input sanitization
- Rate limiting
- OWASP Top 10 protections
- Request validation

## Storage

- ChromaDB for vector storage
- Ray for distributed computing
- ACID-compliant operations

## Binder Folder
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TheMaker369/Hermes/HEAD)