# Artrack API

Track management and collaboration service for the Arkturian platform.

## Features

- 🔐 User Authentication & Authorization
- 📍 Track Management (create, read, update, delete)
- 📌 Waypoints & GPS Routes
- 👥 Collaboration (invitations, sharing)
- 🎯 Snap to Road
- 🎙️ Audio Guides (TTS)
- 📊 Track Analytics

## Tech Stack

- FastAPI
- SQLAlchemy (SQLite/PostgreSQL)
- JWT Authentication
- ElevenLabs TTS
- OpenAI

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your values

# Run migrations
alembic upgrade head

# Start server
uvicorn main:app --host 0.0.0.0 --port 8001
```

## API Documentation

Once running, visit:
- OpenAPI docs: http://localhost:8001/docs
- ReDoc: http://localhost:8001/redoc

## Environment Variables

See `.env.example` for required configuration.

## Dependencies

This service depends on:
- **storage-api** for media object storage (optional)

## Development

```bash
# Run locally
python main.py

# Run tests
pytest

# Format code
black .
```
# Test deployment setup
