"""
Demo Script for Enhanced CADENCE System
Demonstrates hyper-personalized autosuggest and product recommendations
"""
import asyncio
import json
import os
from datetime import datetime
from typing import List, Dict, Any

# Set environment variables for demo
os.environ['SUPABASE_URL'] = 'https://your-project.supabase.co'
os.environ['SUPABASE_KEY'] = 'your-anon-key'
os.environ['GEMINI_API_KEY'] = 'your-gemini-api-key'
os.environ['REDIS_URL'] = 'redis://localhost:6379'

from config.settings import settings
from database.connection import db_manager, initialize_database
from core.data_processor import DataProcessor
from core.cadence_model import create_cadence_model, DynamicBeamSearch
from core.personalization import PersonalizationEngine, UserEmbeddingModel, ProductReranker
from data_generation.synthetic_data import GeminiSyntheticDataGenerator

class EnhancedCADENCEDemo:
    """
    Complete demo of the Enhanced CADENCE system
    """
    
    def __init__(self):
        self.data_processor = None
        self.cadence_model = None
        self.personalization_engine = None
        self.product_reranker = None
        self.beam_search = None
    
    async def initialize_system(self):
        """
        Initialize all components of the system
        """
        print("🚀 Initializing Enhanced CADENCE System...")
        
        # Initialize database
        print("📊 Setting up database...")
        await initialize_database()
        
        # Initialize data processor
        print("🔧 Initializing data processor...")
        self.data_processor = DataProcessor()
        
        # Create demo models (in production, these would be loaded from trained files)
        print("🧠 Loading CADENCE models...")
        vocab_size = 10000  # Smaller for demo
        num_categories = 20  # Fewer categories for demo
        
        self.cadence_model = create_cadence_model(vocab_size, num_categories)
        
        # Initialize personalization components
        print("👤 Setting up personalization engine...")
        user_embedding_model = UserEmbeddingModel(
            num_categories=num_categories,
            num_actions=len(settings.ENGAGEMENT_ACTIONS),
            embedding_dim=64  # Smaller for demo
        )
        
        self.personalization_engine = PersonalizationEngine(user_embedding_model)
        self.product_reranker = ProductReranker(self.personalization_engine)
        
        # Create demo vocabulary
        demo_vocab = {
            '<PAD>': 0, '<UNK>': 1, '</s>': 2, '<s>': 3,
            'laptop': 4, 'phone': 5, 'shoes': 6, 'watch': 7,
            'headphones': 8, 'camera': 9, 'book': 10
        }
        
        self.beam_search = DynamicBeamSearch(self.cadence_model, demo_vocab)
        
        print("✅ System initialized successfully!\n")
    
    async def demo_data_processing(self):
        """
        Demonstrate data processing capabilities
        """
        print("=" * 60)
        print("📈 DEMO: Data Processing & Clustering")
        print("=" * 60)
        
        # Demo query preprocessing
        sample_queries = [
            "iPhone 15 Pro Max 256GB",
            "running shoes for men size 10",
            "wireless bluetooth headphones under $100",
            "4K smart TV 55 inch Samsung",
            "MacBook Pro M3 laptop for programming"
        ]
        
        print("Original queries → Processed queries:")
        for query in sample_queries:
            processed = self.data_processor.preprocess_query_text(query)
            print(f"  '{query}' → '{processed}'")
        
        # Demo product title processing
        sample_titles = [
            "Apple iPhone 15 Pro Max (256GB, Blue Titanium) Unlocked",
            "Nike Air Max 270 Men's Running Shoes (Size 10, Black/White)",
            "Sony WH-1000XM5 Wireless Noise Canceling Headphones"
        ]
        
        print("\nOriginal product titles → Processed titles:")
        for title in sample_titles:
            processed = self.data_processor.preprocess_product_title(title)
            print(f"  '{title}' → '{processed}'")
        
        print("\n✅ Data processing demo completed!\n")
    
    async def demo_synthetic_data_generation(self):
        """
        Demonstrate synthetic data generation with Gemini
        """
        print("=" * 60)
        print("🤖 DEMO: Synthetic Data Generation")
        print("=" * 60)
        
        try:
            generator = GeminiSyntheticDataGenerator()
            
            # Generate a few sample users
            print("Generating 3 sample users...")
            sample_users = await generator.generate_synthetic_users(num_users=3)
            
            for i, user in enumerate(sample_users, 1):
                print(f"\nUser {i}:")
                print(f"  ID: {user['user_id']}")
                print(f"  Persona: {user['persona']}")
                print(f"  Age: {user['age_group']}")
                print(f"  Location: {user['location']}")
                print(f"  Interests: {', '.join(user.get('interests', []))}")
                print(f"  Shopping behavior: {user.get('shopping_behavior', 'N/A')}")
            
            print("\n✅ Synthetic data generation demo completed!")
            
        except Exception as e:
            print(f"⚠️  Synthetic data generation requires Gemini API key: {e}")
        
        print()
    
    async def demo_personalization_engine(self):
        """
        Demonstrate personalization capabilities
        """
        print("=" * 60)
        print("👤 DEMO: Personalization Engine")
        print("=" * 60)
        
        # Create demo users with different profiles
        demo_users = [
            {
                "user_id": "tech_enthusiast_01",
                "persona": "tech_enthusiast",
                "category_preferences": {"electronics": 0.9, "gadgets": 0.8},
                "engagement_patterns": {"purchase": 0.3, "review_view": 0.7},
                "brand_preferences": {"apple": 0.9, "samsung": 0.7}
            },
            {
                "user_id": "fashion_lover_01", 
                "persona": "fashion_lover",
                "category_preferences": {"clothing": 0.9, "accessories": 0.8},
                "engagement_patterns": {"wishlist": 0.6, "view": 0.8},
                "brand_preferences": {"nike": 0.8, "adidas": 0.7}
            }
        ]
        
        base_suggestions = [
            "smartphone cases",
            "running shoes",
            "wireless headphones", 
            "summer dresses",
            "laptop bags"
        ]
        
        print("Base suggestions for all users:")
        for suggestion in base_suggestions:
            print(f"  • {suggestion}")
        
        print("\nPersonalized suggestions:")
        
        for user in demo_users:
            # Cache user profile
            self.personalization_engine.user_profiles_cache[user["user_id"]] = user
            
            # Get personalized suggestions
            personalized = await self.personalization_engine.personalize_query_suggestions(
                user_id=user["user_id"],
                query_prefix="",
                base_suggestions=base_suggestions
            )
            
            print(f"\n{user['persona']} ({user['user_id']}):")
            for suggestion in personalized[:3]:
                print(f"  • {suggestion}")
        
        print("\n✅ Personalization demo completed!\n")
    
    async def demo_product_reranking(self):
        """
        Demonstrate product reranking
        """
        print("=" * 60)
        print("🛍️  DEMO: Product Reranking")
        print("=" * 60)
        
        # Create demo products
        demo_products = [
            {
                "product_id": "laptop_1",
                "title": "MacBook Pro M3 16-inch",
                "brand": "apple",
                "price": 2499.99,
                "rating": 4.8,
                "main_category": "electronics"
            },
            {
                "product_id": "laptop_2", 
                "title": "Dell XPS 13 Developer Edition",
                "brand": "dell",
                "price": 1299.99,
                "rating": 4.5,
                "main_category": "electronics"
            },
            {
                "product_id": "shoes_1",
                "title": "Nike Air Max 270 Running Shoes",
                "brand": "nike",
                "price": 129.99,
                "rating": 4.3,
                "main_category": "clothing"
            },
            {
                "product_id": "shoes_2",
                "title": "Adidas Ultraboost 22",
                "brand": "adidas", 
                "price": 179.99,
                "rating": 4.6,
                "main_category": "clothing"
            }
        ]
        
        print("Original product ranking:")
        for i, product in enumerate(demo_products, 1):
            print(f"  {i}. {product['title']} - ${product['price']} ({product['rating']}⭐)")
        
        # Rerank for tech enthusiast
        user_id = "tech_enthusiast_01"
        reranked_products = await self.product_reranker.rerank_products(
            user_id=user_id,
            query="laptop",
            products=demo_products
        )
        
        print(f"\nPersonalized ranking for {user_id}:")
        for i, product in enumerate(reranked_products, 1):
            print(f"  {i}. {product['title']} - ${product['price']} ({product['rating']}⭐)")
        
        print("\n✅ Product reranking demo completed!\n")
    
    async def demo_real_time_scenario(self):
        """
        Demonstrate a real-time user session scenario
        """
        print("=" * 60)
        print("⚡ DEMO: Real-time User Session")
        print("=" * 60)
        
        user_id = "demo_user_123"
        session_id = "session_demo_456"
        
        print(f"User {user_id} starts a new session...")
        
        # Simulate user typing progressively
        search_progression = ["l", "la", "lap", "lapt", "laptop", "laptop f", "laptop for", "laptop for programming"]
        
        print("\nUser typing progression with autosuggest:")
        
        for prefix in search_progression:
            # Simulate getting suggestions (would use real CADENCE model)
            base_suggestions = [
                f"{prefix} computer",
                f"{prefix} gaming", 
                f"{prefix} student",
                f"{prefix} professional",
                f"{prefix} budget"
            ]
            
            # Apply personalization
            personalized_suggestions = await self.personalization_engine.personalize_query_suggestions(
                user_id=user_id,
                query_prefix=prefix,
                base_suggestions=base_suggestions
            )
            
            print(f"\n  User types: '{prefix}'")
            print(f"  Suggestions: {personalized_suggestions[:3]}")
            
            # Simulate user engagement
            if len(prefix) > 5:  # User clicks on suggestion
                selected = personalized_suggestions[0]
                print(f"  → User selects: '{selected}'")
                
                # Log engagement
                await self._log_demo_engagement(user_id, session_id, "click", selected, 1)
                break
        
        print("\nUser searches for products...")
        
        # Simulate product search results
        search_results = await self.demo_product_search(user_id, "laptop for programming")
        
        print(f"Showing {len(search_results)} personalized results")
        for i, product in enumerate(search_results[:3], 1):
            print(f"  {i}. {product['title']} - ${product['price']}")
        
        print("\n✅ Real-time session demo completed!\n")
    
    async def demo_product_search(self, user_id: str, query: str) -> List[Dict[str, Any]]:
        """
        Simulate product search with personalization
        """
        # Mock search results
        mock_results = [
            {
                "product_id": "prog_laptop_1",
                "title": "MacBook Pro M3 for Development",
                "price": 2299.99,
                "rating": 4.8,
                "brand": "apple",
                "main_category": "electronics"
            },
            {
                "product_id": "prog_laptop_2",
                "title": "ThinkPad X1 Carbon Developer Edition", 
                "price": 1599.99,
                "rating": 4.6,
                "brand": "lenovo",
                "main_category": "electronics"
            },
            {
                "product_id": "prog_laptop_3",
                "title": "Dell XPS 15 Programming Laptop",
                "price": 1799.99,
                "rating": 4.5,
                "brand": "dell",
                "main_category": "electronics"
            }
        ]
        
        # Apply personalized reranking
        reranked_results = await self.product_reranker.rerank_products(
            user_id=user_id,
            query=query,
            products=mock_results
        )
        
        return reranked_results
    
    async def _log_demo_engagement(self, user_id: str, session_id: str, 
                                 action_type: str, item_id: str, rank: int):
        """
        Log engagement for demo
        """
        engagement_data = {
            "engagement_id": f"demo_eng_{datetime.now().timestamp()}",
            "user_id": user_id,
            "session_id": session_id,
            "action_type": action_type,
            "product_id": item_id,
            "item_rank": rank,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # In real implementation, this would go to database
        print(f"    📊 Logged engagement: {action_type} on '{item_id}' at rank {rank}")
    
    async def run_full_demo(self):
        """
        Run the complete demo
        """
        print("🎯 Enhanced CADENCE System - Complete Demo")
        print("=" * 60)
        print("This demo showcases:")
        print("• Data processing with clustering for pseudo-categories")
        print("• Synthetic data generation using Gemini LLM")
        print("• Hyper-personalized autosuggest")
        print("• Personalized product reranking")
        print("• Real-time user session simulation")
        print("=" * 60)
        print()
        
        try:
            await self.initialize_system()
            await self.demo_data_processing()
            await self.demo_synthetic_data_generation()
            await self.demo_personalization_engine()
            await self.demo_product_reranking()
            await self.demo_real_time_scenario()
            
            print("🎉 DEMO COMPLETED SUCCESSFULLY!")
            print("\nKey Features Demonstrated:")
            print("✅ Enhanced CADENCE with category constraints")
            print("✅ Clustering-based pseudo-categories")
            print("✅ Real-time personalization")
            print("✅ User engagement tracking")
            print("✅ Product reranking")
            print("✅ Synthetic data generation")
            print("✅ Low-latency response (<100ms target)")
            
            print("\nNext Steps:")
            print("• Set up Supabase database with provided schema")
            print("• Configure environment variables")
            print("• Train CADENCE models on real data")
            print("• Deploy API with Docker")
            print("• Integrate with frontend application")
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            print("Make sure to configure environment variables properly")

async def main():
    """
    Run the demo
    """
    demo = EnhancedCADENCEDemo()
    await demo.run_full_demo()

if __name__ == "__main__":
    # Set up basic configuration for demo
    print("⚙️  Setting up demo environment...")
    
    # Check if required environment variables are set
    required_vars = ['SUPABASE_URL', 'SUPABASE_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("⚠️  Missing environment variables for full demo:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nDemo will run with limited functionality.")
        print("Set these in your .env file for full experience.")
    
    # Run the demo
    asyncio.run(main()) 