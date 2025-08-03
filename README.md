# 🚀 Enhanced CADENCE System

A state-of-the-art hyper-personalized search autosuggest and product recommendation platform that integrates advanced sequence generation models with robust personalization engines. Built with modern AI/ML technologies and designed for production-scale e-commerce applications.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![React](https://img.shields.io/badge/React-19.1+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![TypeScript](https://img.shields.io/badge/TypeScript-4.9+-blue.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Frontend](#frontend)
- [Model Training](#model-training)
- [Configuration](#configuration)
- [Development](#development)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

The Enhanced CADENCE System is a comprehensive AI-powered search and recommendation platform that provides:

- **Intelligent Autosuggest**: Context-aware query completion using advanced sequence generation
- **Personalized Search**: User-specific product recommendations based on engagement history
- **Real-time Processing**: Fast response times with optimized caching and indexing
- **Scalable Architecture**: Microservices-based design for high availability
- **Modern UI/UX**: Responsive React frontend with Material-UI components

## ✨ Features

### 🤖 AI/ML Capabilities
- **CADENCE Model**: Advanced sequence generation for query autosuggest
- **Dynamic Beam Search**: Configurable beam width for optimal suggestion generation
- **Personalization Engine**: User embedding-based recommendation refinement
- **Engagement Tracking**: Real-time user behavior analysis and learning
- **A/B Testing**: Built-in feature flags for experimentation

### 🔍 Search & Recommendation
- **Smart Autocomplete**: Context-aware query suggestions
- **Product Search**: Advanced filtering and sorting capabilities
- **Category-based Filtering**: Hierarchical product categorization
- **Relevance Scoring**: Multi-factor ranking algorithms
- **Session Management**: Persistent user context across interactions

### 🛠️ Technical Features
- **RESTful API**: Comprehensive FastAPI backend with OpenAPI documentation
- **Real-time Updates**: WebSocket support for live data synchronization
- **Caching Layer**: Redis-based caching for improved performance
- **Database Flexibility**: SQLite (default) with support for PostgreSQL/MySQL
- **Monitoring**: Built-in health checks and system statistics

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend │    │   FastAPI Backend│    │   CADENCE Model │
│   (TypeScript)   │◄──►│   (Python)      │◄──►│   (PyTorch)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         └─────────────►│   Redis Cache   │◄─────────────┘
                        └─────────────────┘
                                │
                        ┌─────────────────┐
                        │   SQLite DB     │
                        └─────────────────┘
```

### Core Components

1. **Frontend Layer** (`frontend/`)
   - React 19+ with TypeScript
   - Material-UI components
   - Real-time autosuggest interface
   - Product search and filtering

2. **API Layer** (`api/`)
   - FastAPI application with CORS support
   - RESTful endpoints for all operations
   - Request/response validation with Pydantic
   - Comprehensive error handling

3. **Core Engine** (`core/`)
   - CADENCE model implementation
   - Personalization algorithms
   - Data processing utilities
   - E-commerce specific logic

4. **Configuration** (`config/`)
   - Environment-based settings
   - Model hyperparameters
   - Feature flags and A/B testing
   - Database and cache configurations

## 🛠️ Technology Stack

### Backend
- **Python 3.8+**: Core programming language
- **FastAPI**: Modern, fast web framework
- **PyTorch 2.0+**: Deep learning framework
- **Transformers**: Hugging Face model library
- **Pydantic**: Data validation and settings
- **SQLite/PostgreSQL**: Database options
- **Redis**: Caching and session management

### Frontend
- **React 19+**: Modern UI framework
- **TypeScript**: Type-safe development
- **Material-UI**: Component library
- **Axios**: HTTP client
- **React Scripts**: Development tooling

### AI/ML
- **CADENCE Model**: Custom sequence generation
- **Sentence Transformers**: Text embeddings
- **Scikit-learn**: Traditional ML algorithms
- **NLTK**: Natural language processing
- **UMAP + HDBSCAN**: Clustering algorithms

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Node.js 16+ and npm
- Redis (optional, for caching)
- Git

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/enhanced-cadence-system.git
   cd enhanced-cadence-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start development server**
   ```bash
   npm start
   ```

## 🏃 Quick Start

### Option 1: Using Batch Files (Windows)

1. **Launch the complete system**
   ```bash
   LAUNCH_CADENCE.bat
   ```

2. **Start training process**
   ```bash
   start_training.bat
   ```

### Option 2: Manual Launch

1. **Start the backend**
   ```bash
   python cadence_backend.py
   # or
   uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Start the frontend**
   ```bash
   cd frontend
   npm start
   ```

3. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## 📚 API Documentation

### Core Endpoints

| Endpoint | Method | Description | Request Body |
|----------|--------|-------------|--------------|
| `/api/v1/autocomplete` | POST | Generate query suggestions | `{"query": "string", "max_suggestions": int}` |
| `/api/v1/search` | POST | Search products | `{"query": "string", "max_results": int}` |
| `/api/v1/categories` | GET | Get available categories | - |
| `/api/v1/stats` | GET | System statistics | - |
| `/health` | GET | Health check | - |

### Example API Usage

```python
import requests

# Autocomplete request
response = requests.post("http://localhost:8000/api/v1/autocomplete", 
    json={"query": "smartphone", "max_suggestions": 5})
suggestions = response.json()["suggestions"]

# Product search
response = requests.post("http://localhost:8000/api/v1/search",
    json={"query": "gaming laptop", "max_results": 10})
products = response.json()["results"]
```

## 🎨 Frontend

The React frontend provides an intuitive interface for:

- **Real-time Autosuggest**: Dynamic query completion as you type
- **Product Search**: Advanced filtering and sorting options
- **User Engagement**: Click tracking and personalization
- **Responsive Design**: Works on desktop and mobile devices

### Key Components

- **Autosuggest Component**: Real-time query suggestions
- **Product Grid**: Display search results with filtering
- **Category Filter**: Hierarchical product categorization
- **User Session**: Persistent user context and preferences

## 🧠 Model Training

### Training the CADENCE Model

1. **Prepare training data**
   ```bash
   python data_generation/generate_synthetic_data.py
   ```

2. **Start training process**
   ```bash
   python fast_cadence_training.py
   # or
   python run_complete_cadence_system.py
   ```

3. **Monitor training progress**
   - Check logs in `training/` directory
   - View metrics and model performance
   - Validate model outputs

### Model Configuration

Key hyperparameters in `config/settings.py`:

```python
# Model Architecture
VOCAB_SIZE = 10000
HIDDEN_DIMS = 512
ATTENTION_DIMS = 64
DROPOUT_RATE = 0.1

# Training Parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100

# Beam Search
BEAM_WIDTH = 5
MAX_SEQUENCE_LENGTH = 50
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with the following variables:

```bash
# Database
DATABASE_URL=sqlite:///cadence.db
REDIS_URL=redis://localhost:6379

# Model Settings
VOCAB_SIZE=10000
BEAM_WIDTH=5
HIDDEN_DIMS=512

# Feature Flags
ENABLE_PERSONALIZATION=true
ENABLE_AB_TESTING=false
ENABLE_TIME_DECAY=true

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=["http://localhost:3000"]
```

### Feature Flags

The system supports various feature flags for A/B testing:

- `ENABLE_PERSONALIZATION`: User-specific recommendations
- `ENABLE_AB_TESTING`: A/B testing framework
- `ENABLE_TIME_DECAY`: Time-based relevance scoring
- `ENABLE_LOCATION_RANKING`: Geographic personalization

## 🧪 Development

### Project Structure

```
enhanced-cadence-system/
├── api/                    # FastAPI application
│   └── main.py            # Main API server
├── core/                   # Core engine modules
│   ├── cadence_model.py   # CADENCE model implementation
│   ├── personalization.py # Personalization algorithms
│   ├── data_processor.py  # Data processing utilities
│   └── ecommerce_autocomplete.py # E-commerce logic
├── frontend/              # React application
│   ├── src/               # Source code
│   ├── public/            # Static assets
│   └── package.json       # Dependencies
├── config/                # Configuration
│   └── settings.py        # Application settings
├── database/              # Database models and migrations
├── data_generation/       # Synthetic data generation
├── training/              # Model training scripts
├── tests/                 # Test suite
├── data/                  # Data files
├── cadence_backend.py     # Main backend entry point
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

### Development Workflow

1. **Set up development environment**
   ```bash
   git clone <repository>
   cd enhanced-cadence-system
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run tests**
   ```bash
   python -m pytest tests/
   ```

3. **Code formatting**
   ```bash
   black .
   isort .
   ```

4. **Type checking**
   ```bash
   mypy .
   ```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_api.py

# Run with coverage
python -m pytest --cov=.

# Run integration tests
python -m pytest tests/integration/
```

### Test Structure

- `tests/unit/`: Unit tests for individual components
- `tests/integration/`: Integration tests for API endpoints
- `tests/model/`: Model training and inference tests
- `tests/frontend/`: Frontend component tests

## 🚀 Deployment

### Production Deployment

1. **Backend Deployment**
   ```bash
   # Build Docker image
   docker build -t cadence-backend .
   
   # Run container
   docker run -p 8000:8000 cadence-backend
   ```

2. **Frontend Deployment**
   ```bash
   cd frontend
   npm run build
   # Deploy build/ directory to your web server
   ```

3. **Database Setup**
   ```bash
   # For PostgreSQL
   pip install psycopg2-binary
   # Update DATABASE_URL in .env
   ```

### Docker Support

Create a `Dockerfile` for containerized deployment:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Add tests for new functionality**
5. **Run the test suite**
   ```bash
   python -m pytest
   ```
6. **Submit a pull request**

### Contribution Guidelines

- Follow PEP 8 for Python code
- Use TypeScript for frontend code
- Add comprehensive tests
- Update documentation as needed
- Ensure all tests pass before submitting

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install

# Run linting
flake8 .
black .
isort .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face**: For the transformers library
- **FastAPI**: For the excellent web framework
- **Material-UI**: For the React component library
- **PyTorch**: For the deep learning framework

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/enhanced-cadence-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/enhanced-cadence-system/discussions)
- **Documentation**: [Wiki](https://github.com/yourusername/enhanced-cadence-system/wiki)

---

**Made with ❤️ by the CADENCE Team**