"""
Main FastAPI Application for Enhanced CADENCE System
Provides real-time hyper-personalized autosuggest and product recommendations
"""
import asyncio
import time
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import structlog

from config.settings import settings
from database.connection import db_manager, initialize_database
from core.data_processor import DataProcessor
from core.cadence_model import CADENCEModel, DynamicBeamSearch, create_cadence_model
from core.personalization import PersonalizationEngine, UserEmbeddingModel, ProductReranker
from training.train_models import CADENCETrainer

logger = structlog.get_logger()

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced CADENCE API",
    description="Hyper-personalized search autosuggest and product recommendations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve React build for frontend
from pathlib import Path
build_dir = Path(__file__).parent.parent / "frontend" / "build"
app.mount("/", StaticFiles(directory=str(build_dir), html=True), name="frontend")

# Global variables for models
cadence_model: Optional[CADENCEModel] = None
personalization_engine: Optional[PersonalizationEngine] = None
product_reranker: Optional[ProductReranker] = None
beam_search: Optional[DynamicBeamSearch] = None
data_processor: Optional[DataProcessor] = None

# Request/Response Models
class AutosuggestRequest(BaseModel):
    query_prefix: str = Field(..., description="The prefix user has typed")
    user_id: str = Field(..., description="Unique user identifier")
    session_id: Optional[str] = Field(None, description="Current session ID")
    max_suggestions: int = Field(10, description="Maximum number of suggestions")
    include_personalization: bool = Field(True, description="Whether to apply personalization")

class AutosuggestResponse(BaseModel):
    suggestions: List[str] = Field(..., description="List of query suggestions")
    personalized: bool = Field(..., description="Whether personalization was applied")
    response_time_ms: float = Field(..., description="Response time in milliseconds")
    session_id: str = Field(..., description="Session ID for tracking")

class ProductSearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    user_id: str = Field(..., description="Unique user identifier")
    session_id: Optional[str] = Field(None, description="Current session ID")
    max_results: int = Field(20, description="Maximum number of products")
    include_personalization: bool = Field(True, description="Whether to apply personalization")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Search filters")

class Product(BaseModel):
    product_id: str
    title: str
    description: Optional[str] = None
    price: Optional[float] = None
    rating: Optional[float] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    image_url: Optional[str] = None

class ProductSearchResponse(BaseModel):
    products: List[Product] = Field(..., description="List of products")
    total_results: int = Field(..., description="Total number of results")
    personalized: bool = Field(..., description="Whether personalization was applied")
    response_time_ms: float = Field(..., description="Response time in milliseconds")
    session_id: str = Field(..., description="Session ID for tracking")

class EngagementEvent(BaseModel):
    user_id: str = Field(..., description="User who performed the action")
    session_id: str = Field(..., description="Current session")
    action_type: str = Field(..., description="Type of engagement action")
    item_id: Optional[str] = Field(None, description="Product/query ID if applicable")
    item_rank: Optional[int] = Field(None, description="Position in list")
    duration_seconds: Optional[float] = Field(None, description="Time spent")
    additional_data: Dict[str, Any] = Field(default_factory=dict, description="Additional context")

class EngagementResponse(BaseModel):
    success: bool = Field(..., description="Whether the event was logged successfully")
    message: str = Field(..., description="Response message")

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize models and database connections"""
    global cadence_model, personalization_engine, product_reranker, beam_search, data_processor
    
    logger.info("Starting Enhanced CADENCE API")
    
    try:
        # Initialize database
        await initialize_database()
        
        # Initialize data processor
        data_processor = DataProcessor()
        
        # Initialize trainer
        trainer = CADENCETrainer()
        
        # Try to load existing trained models
        try:
            logger.info("Loading trained models...")
            cadence_model, vocab, config = trainer.load_model_and_vocab('cadence_trained')
            num_categories = config['num_categories']
            logger.info("Loaded pre-trained CADENCE models")
        except FileNotFoundError:
            logger.warning("⚠️ No pre-trained models found. Running in fallback mode.")
            cadence_model = None
            vocab = {}
            num_categories = 10  # Default categories
        
        # Initialize personalization components
        user_embedding_model = UserEmbeddingModel(
            num_categories=num_categories,
            num_actions=len(settings.ENGAGEMENT_ACTIONS),
            embedding_dim=128
        )
        
        personalization_engine = PersonalizationEngine(user_embedding_model)
        product_reranker = ProductReranker(personalization_engine)
        
        # Initialize beam search
        beam_search = DynamicBeamSearch(cadence_model, vocab)
        
        logger.info("Enhanced CADENCE API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Enhanced CADENCE API")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": cadence_model is not None
    }

# Core API endpoints
@app.post("/autosuggest", response_model=AutosuggestResponse)
async def get_autosuggest(request: AutosuggestRequest, background_tasks: BackgroundTasks):
    """
    Get hyper-personalized autosuggest suggestions
    """
    start_time = time.time()
    
    try:
        # Generate session ID if not provided
        session_id = request.session_id or f"session_{uuid.uuid4().hex[:12]}"
        
        # Step 1: Get base suggestions from CADENCE model
        base_suggestions = await _get_base_suggestions(request.query_prefix)
        
        # Step 2: Apply personalization if requested and user exists
        if request.include_personalization:
            personalized_suggestions = await personalization_engine.personalize_query_suggestions(
                user_id=request.user_id,
                query_prefix=request.query_prefix,
                base_suggestions=base_suggestions
            )
        else:
            personalized_suggestions = base_suggestions
        
        # Step 3: Limit results
        final_suggestions = personalized_suggestions[:request.max_suggestions]
        
        # Step 4: Log the query for future personalization
        background_tasks.add_task(
            _log_autosuggest_query,
            request.user_id,
            session_id,
            request.query_prefix,
            final_suggestions
        )
        
        response_time = (time.time() - start_time) * 1000
        
        return AutosuggestResponse(
            suggestions=final_suggestions,
            personalized=request.include_personalization,
            response_time_ms=response_time,
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"Error in autosuggest: {e}")
        # Fallback to basic suggestions
        response_time = (time.time() - start_time) * 1000
        return AutosuggestResponse(
            suggestions=[f"{request.query_prefix} {suffix}" for suffix in ["", "online", "cheap", "best", "reviews"]],
            personalized=False,
            response_time_ms=response_time,
            session_id=request.session_id or f"session_{uuid.uuid4().hex[:12]}"
        )

@app.post("/search", response_model=ProductSearchResponse)
async def search_products(request: ProductSearchRequest, background_tasks: BackgroundTasks):
    """
    Search for products with hyper-personalized ranking
    """
    start_time = time.time()
    
    try:
        # Generate session ID if not provided
        session_id = request.session_id or f"session_{uuid.uuid4().hex[:12]}"
        
        # Step 1: Get base search results (would integrate with product search engine)
        base_products = await _get_base_search_results(request.query, request.max_results)
        
        # Step 2: Apply personalized reranking if requested
        if request.include_personalization and product_reranker:
            personalized_products = await product_reranker.rerank_products(
                user_id=request.user_id,
                query=request.query,
                products=base_products
            )
        else:
            personalized_products = base_products
        
        # Step 3: Convert to response format
        product_results = [
            Product(
                product_id=p.get('product_id', ''),
                title=p.get('title', ''),
                description=p.get('description'),
                price=p.get('price'),
                rating=p.get('rating'),
                brand=p.get('brand'),
                category=p.get('main_category'),
                image_url=p.get('image_url')
            )
            for p in personalized_products
        ]
        
        # Step 4: Log the search for future personalization
        background_tasks.add_task(
            _log_product_search,
            request.user_id,
            session_id,
            request.query,
            len(product_results)
        )
        
        response_time = (time.time() - start_time) * 1000
        
        return ProductSearchResponse(
            products=product_results,
            total_results=len(product_results),
            personalized=request.include_personalization,
            response_time_ms=response_time,
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"Error in product search: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/engagement", response_model=EngagementResponse)
async def log_engagement(event: EngagementEvent):
    """
    Log user engagement event for personalization
    """
    try:
        # Create engagement record
        engagement_data = {
            "engagement_id": f"eng_{uuid.uuid4().hex[:12]}",
            "user_id": event.user_id,
            "session_id": event.session_id,
            "action_type": event.action_type,
            "product_id": event.item_id,
            "item_rank": event.item_rank,
            "timestamp": datetime.utcnow().isoformat(),
            "duration_seconds": event.duration_seconds
        }
        
        # Save to database
        success = await db_manager.log_engagement(engagement_data)
        
        if success:
            return EngagementResponse(
                success=True,
                message="Engagement logged successfully"
            )
        else:
            return EngagementResponse(
                success=False,
                message="Failed to log engagement"
            )
            
    except Exception as e:
        logger.error(f"Error logging engagement: {e}")
        return EngagementResponse(
            success=False,
            message=f"Error: {str(e)}"
        )

@app.post("/user/session/start")
async def start_user_session(user_id: str, device_type: str = "web"):
    """
    Start a new user session
    """
    try:
        session_id = f"session_{uuid.uuid4().hex[:12]}"
        
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "device_type": device_type,
            "start_time": datetime.utcnow().isoformat()
        }
        
        success = await db_manager.create_session(session_data)
        
        if success:
            return {"session_id": session_id, "message": "Session started successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to start session")
            
    except Exception as e:
        logger.error(f"Error starting session: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/analytics/user/{user_id}")
async def get_user_analytics(user_id: str):
    """
    Get user behavior analytics
    """
    try:
        # Get user profile
        profile = await personalization_engine.get_user_profile(user_id)
        
        # Get recent engagement data
        engagements = await db_manager.get_user_engagements(user_id, limit=100)
        
        # Calculate analytics
        analytics = {
            "user_id": user_id,
            "profile": profile,
            "recent_activity": {
                "total_engagements": len(engagements),
                "engagement_breakdown": _calculate_engagement_breakdown(engagements),
                "last_active": engagements[0]['timestamp'] if engagements else None
            }
        }
        
        return analytics
        
    except Exception as e:
        logger.error(f"Error getting user analytics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Administrative endpoints
@app.post("/admin/retrain-models")
async def retrain_models(max_samples: int = 5000, epochs: int = 1):
    """
    Retrain CADENCE models with new data (admin only)
    """
    try:
        global cadence_model, beam_search
        
        trainer = CADENCETrainer()
        
        # Train new models
        new_model, new_vocab, cluster_mappings = trainer.train_full_pipeline(
            max_samples=max_samples,
            epochs=epochs
        )
        
        # Update global models
        cadence_model = new_model
        beam_search = DynamicBeamSearch(cadence_model, new_vocab)
        
        return {
            "message": f"Successfully retrained models with {max_samples} samples",
            "vocab_size": len(new_vocab),
            "num_categories": len(cluster_mappings.get('query_clusters', {})) + len(cluster_mappings.get('product_clusters', {}))
        }
            
    except Exception as e:
        logger.error(f"Error retraining models: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/admin/stats")
async def get_system_stats():
    """
    Get system statistics
    """
    try:
        # This would query the database for stats
        stats = {
            "total_users": 0,  # Would be queried from DB
            "total_sessions": 0,
            "total_queries": 0,
            "total_engagements": 0,
            "models_loaded": {
                "cadence_model": cadence_model is not None,
                "personalization_engine": personalization_engine is not None,
                "product_reranker": product_reranker is not None
            }
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Helper functions
async def _get_base_suggestions(query_prefix: str) -> List[str]:
    """
    Get base suggestions from CADENCE model
    """
    try:
        if not beam_search or not cadence_model:
            # Fallback suggestions
            return [
                f"{query_prefix} for sale",
                f"{query_prefix} online", 
                f"{query_prefix} cheap",
                f"{query_prefix} best",
                f"{query_prefix} reviews"
            ]
        
        # Try to use real CADENCE model for generation
        try:
            # Simple prefix completion for demo
            common_completions = {
                'l': ['laptop gaming', 'laptop programming', 'laptop student', 'laptop business'],
                'la': ['laptop macbook', 'laptop dell', 'laptop hp', 'laptop lenovo'],
                'lap': ['laptop computer', 'laptop backpack', 'laptop stand', 'laptop cooling pad'],
                'lapt': ['laptop gaming', 'laptop ultrabook', 'laptop convertible'],
                'laptop': ['laptop for programming', 'laptop for gaming', 'laptop for students', 'laptop for business'],
                'p': ['phone case', 'phone charger', 'phone samsung', 'phone apple'],
                'ph': ['phone accessories', 'phone holder', 'phone screen protector'],
                'pho': ['phone case', 'phone charger', 'phone wireless', 'phone bluetooth'],
                'phon': ['phone case protective', 'phone charger fast', 'phone stand', 'phone mount'],
                'phone': ['phone case', 'phone charger', 'phone accessories', 'phone screen protector'],
                'h': ['headphones wireless', 'headphones bluetooth', 'headphones gaming', 'headphones noise cancelling'],
                'he': ['headphones sony', 'headphones apple', 'headphones bose'],
                'hea': ['headphones wireless', 'headphones over ear', 'headphones in ear'],
                'head': ['headphones bluetooth', 'headphones gaming', 'headphones studio'],
                'headp': ['headphones wireless', 'headphones noise cancelling'],
                'headph': ['headphones bluetooth', 'headphones wireless'],
                'headpho': ['headphones wireless bluetooth', 'headphones noise cancelling'],
                'headphon': ['headphones wireless', 'headphones bluetooth'],
                'headphone': ['headphones wireless', 'headphones bluetooth', 'headphones gaming'],
                'headphones': ['headphones wireless', 'headphones bluetooth', 'headphones noise cancelling', 'headphones gaming'],
                's': ['shoes running', 'smartphone', 'smartwatch', 'speaker bluetooth'],
                'sh': ['shoes nike', 'shoes adidas', 'shirt', 'shorts'],
                'sho': ['shoes running', 'shoes casual', 'shoes sports', 'shoes formal'],
                'shoe': ['shoes for men', 'shoes for women', 'shoes running', 'shoes casual'],
                'shoes': ['shoes running', 'shoes casual', 'shoes sports', 'shoes formal'],
                'w': ['watch smart', 'wireless headphones', 'water bottle', 'wallet'],
                'wa': ['watch apple', 'watch samsung', 'wallet leather', 'water bottle'],
                'wat': ['watch smartwatch', 'water bottle steel', 'watch digital'],
                'watc': ['watch smart', 'watch fitness', 'watch analog'],
                'watch': ['watch smart', 'watch fitness', 'watch analog', 'watch digital']
            }
            
            suggestions = common_completions.get(query_prefix.lower(), [
                f"{query_prefix} best",
                f"{query_prefix} online", 
                f"{query_prefix} cheap",
                f"{query_prefix} quality",
                f"{query_prefix} sale"
            ])
            
            return suggestions[:5]  # Return top 5
            
        except Exception as model_error:
            logger.warning(f"Model generation failed: {model_error}")
            # Fallback to simple suggestions
            return [
                f"{query_prefix} best",
                f"{query_prefix} cheap",
                f"{query_prefix} online",
                f"{query_prefix} sale",
                f"{query_prefix} quality"
            ]
        
    except Exception as e:
        logger.error(f"Error getting base suggestions: {e}")
        return [f"{query_prefix} {suffix}" for suffix in ["best", "online", "cheap", "sale", "quality"]]

async def _get_base_search_results(query: str, max_results: int) -> List[Dict[str, Any]]:
    """
    Get base search results (would integrate with product search engine)
    """
    try:
        # This would integrate with your product search engine
        # For demo, return mock products
        base_products = []
        
        for i in range(min(max_results, 20)):
            product = {
                'product_id': f'prod_{i}',
                'title': f'{query} Product {i}',
                'description': f'Description for {query} product {i}',
                'price': 50.0 + (i * 10),
                'rating': 3.5 + (i % 3) * 0.5,
                'brand': f'Brand {i % 5}',
                'main_category': 'electronics',
                'image_url': f'https://example.com/product_{i}.jpg'
            }
            base_products.append(product)
        
        return base_products
        
    except Exception as e:
        logger.error(f"Error getting search results: {e}")
        return []

async def _log_autosuggest_query(user_id: str, session_id: str, query_prefix: str, suggestions: List[str]):
    """
    Log autosuggest query for analytics
    """
    try:
        query_data = {
            "query_id": f"query_{uuid.uuid4().hex[:12]}",
            "session_id": session_id,
            "user_id": user_id,
            "query_text": query_prefix,
            "suggested_completions": suggestions,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await db_manager.log_search_query(query_data)
        
    except Exception as e:
        logger.error(f"Error logging autosuggest query: {e}")

async def _log_product_search(user_id: str, session_id: str, query: str, result_count: int):
    """
    Log product search for analytics
    """
    try:
        query_data = {
            "query_id": f"query_{uuid.uuid4().hex[:12]}",
            "session_id": session_id,
            "user_id": user_id,
            "query_text": query,
            "results_shown": result_count,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await db_manager.log_search_query(query_data)
        
    except Exception as e:
        logger.error(f"Error logging product search: {e}")

def _calculate_engagement_breakdown(engagements: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Calculate breakdown of engagement types
    """
    breakdown = {}
    for engagement in engagements:
        action_type = engagement.get('action_type', 'unknown')
        breakdown[action_type] = breakdown.get(action_type, 0) + 1
    
    return breakdown

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 