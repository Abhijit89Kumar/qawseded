# Enhanced CADENCE System

A hyper-personalized search autosuggest and product recommendation platform integrating a sequence generation model (CADENCE) with a robust personalization engine.

## Architecture Overview

The system comprises:
- **CADENCE Model**: A deep sequence generation architecture for query autosuggest using dynamic beam search.
- **Personalization Engine**: Refines suggestions and search results based on user embeddings and engagement signals.
- **FastAPI Backend**: Exposes REST endpoints for autosuggest, product search, engagement logging, and administrative operations.
- **React Frontend**: A user interface built with React and MUI for real-time interaction.
- **Database & Cache**: SQLite (or configurable via `DATABASE_URL`) and Redis for session management and caching.
- **Configuration**: Centralized in `config/settings.py` using Pydantic BaseSettings.

## CADENCE Model

CADENCE (Contextual Auto-Dependency Encoder for Natural Conversational Entities) generates query suggestions token-by-token:
- **Vocabulary Size**: Configured via `VOCAB_SIZE`.
- **Embedding Backbone**: `sentence-transformers/all-MiniLM-L6-v2` for initial token embeddings.
- **Beam Search**: Dynamic beam width (`BEAM_WIDTH`) to explore top candidate sequences.
- **Model Config**: Hyperparameters (`HIDDEN_DIMS`, `ATTENTION_DIMS`, `DROPOUT_RATE`) are defined in settings.

## Personalization Engine

Enhances raw model outputs using:
- **User Embeddings**: Built from historical engagement (up to `MAX_USER_HISTORY`).
- **Product Re-Ranker**: Applies engagement weights (`FEATURE_FLAGS`, `ENGAGEMENT_WEIGHTS`) to rank suggestions.
- **Engagement Signals**: Actions like `view`, `click`, `add_to_cart`, `purchase`, etc., inform personalization scores.

## API Endpoints

| Path                   | Method | Purpose                                     |
|------------------------|--------|---------------------------------------------|
| `/autosuggest`         | POST   | Generate query suggestions                  |
| `/search`              | POST   | Retrieve personalized product search results|
| `/engagement`          | POST   | Log user engagement events                  |
| `/user/session/start`  | POST   | Start or resume a user session              |
| `/health`              | GET    | Health check                                |
| `/admin/retrain-models`| POST   | Trigger model retraining                    |
| `/admin/stats`         | GET    | Fetch system statistics                     |

## Frontend

Located in `frontend/`, the React app uses MUI components and communicates with the backend:
- **Autosuggest**: Real-time dropdown for query prefixes.
- **Search Results**: Filterable and sortable product listings.
- **Engagement Tracking**: Sends user actions back to the API for personalization.

## Configuration

All settings are in `config/settings.py` (override via `.env`):
- `DATABASE_URL`, `REDIS_URL`
- Model hyperparameters and beam search settings.
- Feature flags: A/B testing, time decay, location-based ranking, etc.

## Directory Structure

```
/api              # FastAPI application code
/frontend         # React web application
/config           # Environment and configuration settings
/database         # ORM models and DB connection logic
/core             # Core modules: model definitions, data processing
/data_generation  # Synthetic data generation scripts
/tests            # Automated tests
/README.md        # This consolidated technical documentation
```

## Getting Started

1. Install backend dependencies: `pip install -r requirements.txt`.
2. Start Redis if enabled or configure `REDIS_URL`.
3. Launch backend: `uvicorn api.main:app --reload`.
4. Launch frontend: `npm install && npm start` in `frontend/`.

## Contributing

Contributions are welcome! Please submit pull requests adhering to code style, and add tests for new functionality.