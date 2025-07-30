#!/usr/bin/env python3
"""
Simple mock backend for CADENCE demo
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import time
import uuid

app = FastAPI(title="CADENCE Mock API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class AutosuggestRequest(BaseModel):
    query_prefix: str
    user_id: str
    session_id: Optional[str] = None
    max_suggestions: int = 10
    include_personalization: bool = True

class AutosuggestResponse(BaseModel):
    suggestions: List[str]
    personalized: bool
    response_time_ms: float
    session_id: str

class ProductSearchRequest(BaseModel):
    query: str
    user_id: str
    session_id: Optional[str] = None
    max_results: int = 20
    include_personalization: bool = True

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
    products: List[Product]
    total_results: int
    personalized: bool
    response_time_ms: float
    session_id: str

class EngagementEvent(BaseModel):
    user_id: str
    session_id: str
    action_type: str
    item_id: Optional[str] = None
    item_rank: Optional[int] = None
    duration_seconds: Optional[float] = None
    additional_data: Dict[str, Any] = {}

class EngagementResponse(BaseModel):
    success: bool
    message: str

# Mock data
SUGGESTIONS_MAP = {
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

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/autosuggest", response_model=AutosuggestResponse)
async def get_autosuggest(request: AutosuggestRequest):
    start_time = time.time()
    
    # Get suggestions based on prefix
    prefix = request.query_prefix.lower().strip()
    suggestions = SUGGESTIONS_MAP.get(prefix, [
        f"{request.query_prefix} best",
        f"{request.query_prefix} online", 
        f"{request.query_prefix} cheap",
        f"{request.query_prefix} quality",
        f"{request.query_prefix} sale"
    ])
    
    # Limit to max_suggestions
    suggestions = suggestions[:request.max_suggestions]
    
    response_time = (time.time() - start_time) * 1000
    session_id = request.session_id or f"session_{uuid.uuid4().hex[:12]}"
    
    return AutosuggestResponse(
        suggestions=suggestions,
        personalized=request.include_personalization,
        response_time_ms=response_time,
        session_id=session_id
    )

@app.post("/search", response_model=ProductSearchResponse)
async def search_products(request: ProductSearchRequest):
    start_time = time.time()
    
    # Generate mock products based on query
    products = []
    for i in range(min(request.max_results, 12)):
        products.append(Product(
            product_id=f"prod_{i}",
            title=f"{request.query.title()} Product {i+1}",
            description=f"High-quality {request.query} with excellent features",
            price=50.0 + (i * 25),
            rating=3.5 + (i % 3) * 0.5,
            brand=f"Brand {i % 5 + 1}",
            category="Electronics",
            image_url=f"https://example.com/product_{i}.jpg"
        ))
    
    response_time = (time.time() - start_time) * 1000
    session_id = request.session_id or f"session_{uuid.uuid4().hex[:12]}"
    
    return ProductSearchResponse(
        products=products,
        total_results=len(products),
        personalized=request.include_personalization,
        response_time_ms=response_time,
        session_id=session_id
    )

@app.post("/engagement", response_model=EngagementResponse)
async def log_engagement(event: EngagementEvent):
    # Mock engagement logging
    return EngagementResponse(
        success=True,
        message="Engagement logged successfully"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 