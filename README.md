# Enhanced CADENCE: Hyper-Personalized E-commerce Search & Recommendations

![Enhanced CADENCE System](https://img.shields.io/badge/Enhanced%20CADENCE-v1.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

> **Revolutionizing e-commerce search with neural query generation, clustering-based categorization, and real-time hyper-personalization.**

## 🎯 Overview

Enhanced CADENCE is a sophisticated evolution of the original CADENCE paper, transforming basic query autocomplete into a comprehensive hyper-personalized search and recommendation system. Built for modern e-commerce platforms like Flipkart, it combines cutting-edge ML techniques with real-time personalization to deliver sub-100ms response times.

### 🔬 Based on Research

This implementation extends the **CADENCE: Offline Category Constrained and Diverse Query Generation for E-commerce Autosuggest** paper (KDD 2023) with:

- **Clustering-based pseudo-categories** instead of predefined taxonomies
- **Real-time hyper-personalization** using user engagement patterns
- **Dual-task architecture** for both autosuggest and product recommendations
- **Synthetic data generation** using Gemini LLM for scalable training

## 🚀 Key Features

### 🧠 Enhanced CADENCE Core
- **GRU-MN Architecture**: Memory networks with self-attention for context understanding
- **Category Constraints**: Prevents concept drift during query generation
- **Dynamic Beam Search**: Generates diverse, relevant suggestions
- **Dual Language Models**: Separate models for queries and product catalogs

### 👤 Hyper-Personalization Engine
- **Real-time User Profiling**: Learns from every user interaction
- **Engagement-based Scoring**: Weights actions by business value (view: 1x, purchase: 10x)
- **Collaborative Filtering**: Leverages similar user behavior patterns
- **Context-aware Ranking**: Considers session context, time patterns, location

### 📊 Advanced Data Processing
- **HDBSCAN Clustering**: Automatic discovery of pseudo-categories
- **Extractive Summarization**: Entropy and PMI-based product title cleaning
- **Multi-modal Embeddings**: Sentence transformers for semantic understanding

### 🤖 Synthetic Data Generation
- **Gemini LLM Integration**: Generates realistic user personas and behaviors
- **Scalable Training Data**: Creates millions of synthetic interactions
- **Diverse User Patterns**: Multiple shopping personas and engagement styles

## 📁 Project Structure

```
Enhanced-CADENCE/
├── 📁 config/
│   └── settings.py              # Configuration and feature flags
├── 📁 database/
│   ├── models.py               # Pydantic models and SQL schema
│   └── connection.py           # Supabase and Redis connection management
├── 📁 core/
│   ├── data_processor.py       # Data preprocessing and clustering
│   ├── cadence_model.py        # GRU-MN models and beam search
│   └── personalization.py     # Personalization engine and rerankers
├── 📁 data_generation/
│   └── synthetic_data.py       # Gemini-powered synthetic data generation
├── 📁 api/
│   └── main.py                 # FastAPI endpoints for real-time serving
├── demo.py                     # Comprehensive system demonstration
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🛠️ Setup & Installation

### Prerequisites

- **Python 3.8+**
- **Git** (to clone the repository)

That's it! No external database setup required - everything runs locally with SQLite.

### 🚀 One-Click Demo (Recommended)

**For any OS:**
```bash
git clone https://github.com/your-username/enhanced-cadence.git
cd enhanced-cadence
python run_demo.py
```

**Windows users can also use:**
```bash
run_demo.bat
```

The demo script automatically:
- ✅ Installs required dependencies
- ✅ Sets up SQLite database  
- ✅ Trains CADENCE models (if needed)
- ✅ Starts the web server at http://localhost:8000
- ✅ Shows interactive demo instructions

### 🔧 Manual Setup (Advanced)

### 1. Clone & Install

```bash
git clone https://github.com/your-username/enhanced-cadence.git
cd enhanced-cadence
pip install -r requirements.txt
```

### 2. Environment Configuration (Optional)

The demo works out-of-the-box, but you can customize settings:

```env
# Database (SQLite for local development)
DATABASE_URL=sqlite:///./cadence.db

# Cache (Optional)
REDIS_URL=redis://localhost:6379

# Model configurations
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
MAX_LATENCY_MS=100
ENABLE_AB_TESTING=true
```

### 3. Database Setup

The system automatically creates SQLite tables:

```sql
-- Core tables for users, sessions, queries, engagements
-- Product catalog with embeddings and clustering
-- A/B testing and analytics tables
-- All stored locally in cadence.db
```

### 4. Manual Start

```bash
# Train models (optional - done automatically)
python training/train_models.py

# Start the API server
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Open browser to http://localhost:8000
```

## 🎯 Core Workflows

### 1. Hyper-Personalized Autosuggest

```python
# User types "wa" in search bar
POST /autosuggest
{
    "query_prefix": "wa",
    "user_id": "user_123",
    "session_id": "session_456",
    "max_suggestions": 10,
    "include_personalization": true
}

# Response (< 100ms)
{
    "suggestions": [
        "washing machine front load",
        "watch for men smart",
        "water bottle steel",
        "wall mount tv stand",
        "wallet leather brown"
    ],
    "personalized": true,
    "response_time_ms": 45,
    "session_id": "session_456"
}
```

### 2. Personalized Product Search

```python
# User searches for products
POST /search
{
    "query": "laptop for programming",
    "user_id": "user_123",
    "max_results": 20,
    "include_personalization": true
}

# Returns reranked products based on:
# - User's historical preferences
# - Similar users' behaviors  
# - Current session context
# - Engagement patterns
```

### 3. Real-time Engagement Tracking

```python
# Track every user action
POST /engagement
{
    "user_id": "user_123",
    "session_id": "session_456", 
    "action_type": "add_to_cart",
    "item_id": "product_789",
    "item_rank": 2,
    "duration_seconds": 30.5
}

# Engagement weights:
# view: 1.0, click: 2.0, add_to_cart: 5.0
# wishlist: 3.0, purchase: 10.0, review_view: 1.5
```

## 🧠 Technical Architecture

### Model Pipeline

```mermaid
graph TD
    A[User Input: "wa"] --> B[CADENCE Models]
    B --> C[Query LM + Catalog LM]
    C --> D[Dynamic Beam Search]
    D --> E[Base Suggestions]
    E --> F[Personalization Engine]
    F --> G[User Profile + Context]
    G --> H[Reranked Suggestions]
    H --> I[< 100ms Response]
    
    J[User Engagement] --> K[Real-time Updates]
    K --> F
```

### Data Flow

1. **Input Processing**: Query preprocessing, clustering, category assignment
2. **Model Inference**: GRU-MN models with category constraints
3. **Candidate Generation**: Dynamic beam search for diversity
4. **Personalization**: Real-time reranking based on user behavior
5. **Response**: Sub-100ms delivery with engagement tracking

### Personalization Scoring

```python
final_score = (
    0.4 * personal_preference_score +      # Historical behavior
    0.3 * contextual_score +               # Current session
    0.2 * collaborative_score +            # Similar users
    0.1 * temporal_score                   # Time patterns
)
```

## 📊 Datasets Used

### Training Data

1. **[Amazon QAC Dataset](https://huggingface.co/datasets/amazon/AmazonQAC)** (395M samples)
   - Real user typing patterns and search completions
   - Session context and temporal information
   - Used for Query Language Model training

2. **[Amazon Products 2023](https://huggingface.co/datasets/milistu/AMAZON-Products-2023)** (117K products)
   - Product titles, descriptions, categories, ratings
   - Pre-computed embeddings for semantic search
   - Used for Catalog Language Model training

### Synthetic Data Generation

- **User Personas**: 14 distinct shopping personas (tech enthusiast, fashion lover, etc.)
- **Engagement Patterns**: 5 behavior types (browser, researcher, impulse buyer, etc.)
- **Session Simulation**: Realistic shopping journeys with progressive queries
- **Scalable**: Generate millions of interactions using Gemini LLM

## 🔥 Performance & Scalability

### Response Times
- **Autosuggest**: < 50ms (target < 100ms)
- **Product Search**: < 200ms (target < 300ms)
- **Engagement Logging**: < 10ms

### Scalability Features
- **Redis Caching**: User profiles and frequent queries
- **Async Processing**: Background model updates
- **Horizontal Scaling**: Stateless API design
- **Database Optimization**: Indexed queries and partitioning

### A/B Testing Framework
- **Real-time Experimentation**: Compare different algorithms
- **Statistical Significance**: Automated test validation
- **Business Metrics**: CTR, conversion, engagement tracking

## 🎮 Demo Features

Run `python demo.py` to see:

```
🎯 Enhanced CADENCE System - Complete Demo
============================================================
This demo showcases:
• Data processing with clustering for pseudo-categories
• Synthetic data generation using Gemini LLM  
• Hyper-personalized autosuggest
• Personalized product reranking
• Real-time user session simulation
============================================================

📈 DEMO: Data Processing & Clustering
'iPhone 15 Pro Max 256GB' → 'iphone 15 pro max 256 gb'
'running shoes for men size 10' → 'running shoe for men size 10'

👤 DEMO: Personalization Engine
Base suggestions: [smartphone cases, running shoes, ...]
tech_enthusiast: [wireless headphones, smartphone cases, laptop bags]
fashion_lover: [running shoes, summer dresses, ...]

⚡ DEMO: Real-time User Session
User types: 'l' → Suggestions: [laptop computer, laptop gaming, ...]
User types: 'laptop' → Suggestions: [laptop for programming, ...]
→ User selects: 'laptop for programming'
📊 Logged engagement: click on 'laptop for programming' at rank 1
```

## 🚀 API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/autosuggest` | POST | Get personalized query suggestions |
| `/search` | POST | Search products with personalized ranking |
| `/engagement` | POST | Log user engagement events |
| `/user/session/start` | POST | Start new user session |

### Analytics & Admin

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analytics/user/{user_id}` | GET | User behavior analytics |
| `/admin/generate-synthetic-data` | POST | Generate training data |
| `/admin/stats` | GET | System performance stats |
| `/health` | GET | Health check |

## 🧪 Evaluation Metrics

### Offline Metrics
- **Model Perplexity**: Language model quality
- **Top-k Accuracy**: Suggestion relevance  
- **Diversity Scores**: Facet coverage and uniqueness
- **Category Drift**: Concept consistency

### Online Metrics
- **AS Usage**: % queries using autosuggest
- **Search CTR**: Click-through rates
- **Conversion Rate**: Purchase completion
- **Engagement Score**: Weighted action values

### Business Impact
- **Cold Start Reduction**: New product visibility (+3.7% in original paper)
- **Query Coverage**: Increased suggestion availability
- **User Satisfaction**: Reduced typing effort
- **Revenue Impact**: Improved conversion rates

## 🛡️ Production Considerations

### Security
- **Row Level Security**: Supabase user isolation
- **API Rate Limiting**: Prevent abuse
- **Data Privacy**: Anonymized user tracking
- **Input Validation**: SQL injection prevention

### Monitoring
- **Performance Metrics**: Response times, error rates
- **Model Drift Detection**: Quality degradation alerts
- **User Behavior Analytics**: Engagement patterns
- **A/B Test Results**: Statistical significance tracking

### Deployment
```yaml
# docker-compose.yml
version: '3.8'
services:
  cadence-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - SUPABASE_URL=${SUPABASE_URL}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

## 🔬 Research Extensions

### Implemented Enhancements
1. **Clustering-based Categories**: HDBSCAN for automatic categorization
2. **Real-time Personalization**: Sub-100ms user-specific ranking
3. **Dual-task Learning**: Unified autosuggest + product ranking
4. **Synthetic Data Pipeline**: Scalable training data generation

### Future Research Directions
1. **Multi-modal Models**: Image + text understanding
2. **Conversational Search**: Natural language queries
3. **Cross-platform Learning**: Mobile vs desktop behaviors
4. **Federated Learning**: Privacy-preserving personalization

## 📚 References

1. **Original CADENCE Paper**: [KDD 2023](https://doi.org/10.1145/3580305.3599787)
2. **Amazon QAC Dataset**: [Hugging Face](https://huggingface.co/datasets/amazon/AmazonQAC)
3. **Amazon Products 2023**: [Hugging Face](https://huggingface.co/datasets/milistu/AMAZON-Products-2023)
4. **HDBSCAN Clustering**: [JMLR 2017](https://jmlr.org/papers/v18/16-535.html)

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎖️ Acknowledgments

- **Original CADENCE Authors**: Abhinav Anand, Surender Kumar, Nandeesh Kumar, Samir Shah
- **Flipkart Research**: For the foundational research and methodology
- **Amazon**: For providing the QAC and Products datasets
- **Google**: For Gemini LLM capabilities for synthetic data generation

---

**🚀 Ready to revolutionize your e-commerce search experience? Get started with Enhanced CADENCE today!**

For questions, issues, or collaboration opportunities, please open an issue or contact the maintainers. 