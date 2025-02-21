# Hermes AI System Quick Start Guide

## 🚀 5-Minute Setup

```bash
# 1. Clone and setup
git clone https://github.com/TheMaker369/Hermes
cd Hermes
python -m venv env
source env/bin/activate  # or `env\Scripts\activate` on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
cp .env.example .env
# Edit .env with your API keys

# 4. Start the system
docker-compose up --build
```

## 🎯 Core Features

### 1. MAGI (Multi-Agent Intelligence)
```python
from hermes.sephirot.chokmah import MAGI

response = await MAGI().process({
    "query": "Analyze the impact of AI on society",
    "depth": 2
})
```

### 2. APRE (Pattern Recognition)
```python
from hermes.sephirot.binah import APRE

patterns = await APRE().analyze({
    "data": "Your text or data here",
    "pattern_type": "semantic"
})
```

### 3. Data Storage
```python
from hermes.sephirot.yesod import Yesod

await Yesod().store({
    "knowledge": "Important insight",
    "metadata": {"source": "analysis"}
})
```

## 🔍 Key Endpoints

1. **Analysis**: `POST /api/v1/analyze`
2. **Pattern Search**: `POST /api/v1/patterns`
3. **Knowledge Store**: `POST /api/v1/store`
4. **Integration**: `POST /api/v1/integrate`

## 📊 System Status

- **Dashboard**: `http://localhost:8265` (Ray)
- **Storage**: `http://localhost:8001` (ChromaDB)
- **API**: `http://localhost:8000` (FastAPI)

## 🛠️ Common Tasks

### 1. Add New Pattern Type
```python
# In hermes/sephirot/binah.py
@pattern_detector
def detect_new_pattern(data: Dict) -> Pattern:
    # Your pattern detection logic
    pass
```

### 2. Create MAGI Agent
```python
# In hermes/sephirot/chokmah.py
@agent
class NewAgent(BaseAgent):
    async def process(self, input: Dict) -> Dict:
        # Your agent logic
        pass
```

### 3. Add External Integration
```python
# In hermes/sephirot/netzach.py
@service_integration
async def new_service(data: Dict) -> Dict:
    # Your integration logic
    pass
```

## 🔐 Security Best Practices

1. **API Keys**: Never commit `.env` file
2. **Rate Limiting**: Configured in `gevurah.py`
3. **Input Validation**: Use Pydantic models
4. **Monitoring**: Check Ray dashboard

## 📚 Documentation Map

1. `docs/ARCHITECTURE.md`: System design
2. `docs/PATTERNS.md`: Code patterns
3. `docs/PHILOSOPHY.md`: Core concepts
4. `SYSTEM_OVERVIEW.py`: Complete overview

## 🆘 Troubleshooting

1. **Services Won't Start**
   ```bash
   docker-compose down -v
   docker-compose up --build
   ```

2. **Memory Issues**
   ```bash
   # Edit docker-compose.yml
   services:
     hermes:
       deploy:
         resources:
           limits:
             memory: 4G
   ```

3. **Pattern Detection Fails**
   ```python
   # Enable debug logging
   import logging
   logging.getLogger('hermes').setLevel(logging.DEBUG)
   ```

## 🎯 Next Steps

1. Read `docs/ARCHITECTURE.md` for deep understanding
2. Explore `docs/PATTERNS.md` for best practices
3. Study `docs/PHILOSOPHY.md` for core concepts
4. Check `SYSTEM_OVERVIEW.py` for complete picture

## 💡 Pro Tips

1. Use Ray dashboard for performance monitoring
2. Enable pattern caching for faster responses
3. Implement circuit breakers for external calls
4. Use ChromaDB collections for organized storage

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Follow patterns in `docs/PATTERNS.md`
4. Submit pull request

Remember: Hermes follows the Tree of Life architecture - each component has its place and purpose. Start simple, then explore deeper as needed.
