#!/usr/bin/env python3
"""
Run Enhanced CADENCE System with Real Data Generation and Training
This script sets up and runs the complete enhanced CADENCE system with:
1. Real Amazon QAC dataset processing
2. Synthetic data generation using Gemini
3. Enhanced model training
4. Real product database integration
"""
import asyncio
import os
import sys
import json
import torch
from pathlib import Path
from datetime import datetime
import structlog

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config.settings import settings
from data_generation.synthetic_data import generate_full_synthetic_dataset
from core.data_processor import DataProcessor
from training.train_models import CADENCETrainer
from database.connection import initialize_database

logger = structlog.get_logger()

async def setup_enhanced_cadence_system():
    """
    Setup the complete enhanced CADENCE system with real data
    """
    logger.info("🚀 Starting Enhanced CADENCE System Setup")
    
    # Step 1: Initialize database
    logger.info("📊 Step 1: Initializing database...")
    try:
        await initialize_database()
        logger.info("✅ Database initialized successfully")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False
    
    # Step 2: Generate synthetic data using Gemini
    logger.info("🤖 Step 2: Generating synthetic data with Gemini LLM...")
    try:
        success = await generate_full_synthetic_dataset(num_users=500)
        if success:
            logger.info("✅ Synthetic data generation completed")
        else:
            logger.warning("⚠️ Synthetic data generation had issues, continuing...")
    except Exception as e:
        logger.error(f"❌ Synthetic data generation failed: {e}")
        logger.info("📝 Continuing without synthetic data...")
    
    # Step 3: Process Amazon datasets
    logger.info("📈 Step 3: Processing Amazon datasets...")
    try:
        data_processor = DataProcessor()
        
        # Process Amazon QAC dataset (larger sample)
        logger.info("Processing Amazon QAC dataset...")
        query_df = data_processor.load_and_process_amazon_qac(max_samples=50000)
        logger.info(f"✅ Processed {len(query_df)} queries from Amazon QAC")
        
        # Process Amazon Products dataset
        logger.info("Processing Amazon Products dataset...")
        product_df = data_processor.load_and_process_amazon_products(max_samples=25000)
        logger.info(f"✅ Processed {len(product_df)} products from Amazon dataset")
        
        # Create training data
        training_data = data_processor.create_training_data(query_df, product_df)
        
        # Save processed data
        processed_data_dir = Path("processed_data")
        processed_data_dir.mkdir(exist_ok=True)
        
        query_df.to_parquet(processed_data_dir / "queries.parquet")
        product_df.to_parquet(processed_data_dir / "products.parquet")
        
        with open(processed_data_dir / "cluster_mappings.json", "w") as f:
            json.dump({
                "query_clusters": training_data["query_clusters"],
                "product_clusters": training_data["product_clusters"],
                "cluster_mapping": training_data["cluster_mapping"]
            }, f, indent=2)
        
        logger.info("✅ Amazon datasets processed and saved")
        
    except Exception as e:
        logger.error(f"❌ Amazon dataset processing failed: {e}")
        return False
    
    # Step 4: Train enhanced CADENCE models
    logger.info("🧠 Step 4: Training enhanced CADENCE models...")
    try:
        trainer = CADENCETrainer()
        
        # Train with the processed data
        model, vocab, cluster_mappings = trainer.train_enhanced_models(
            training_data=training_data,
            epochs=3,
            save_name="enhanced_cadence"
        )
        
        logger.info("✅ Enhanced CADENCE models trained successfully")
        
    except Exception as e:
        logger.error(f"❌ Model training failed: {e}")
        return False
    
    # Step 5: Test the system
    logger.info("🧪 Step 5: Testing the enhanced system...")
    try:
        await test_enhanced_system(model, vocab)
        logger.info("✅ System testing completed")
    except Exception as e:
        logger.error(f"❌ System testing failed: {e}")
    
    logger.info("🎉 Enhanced CADENCE System setup completed successfully!")
    return True

async def test_enhanced_system(model, vocab):
    """Test the enhanced CADENCE system"""
    logger.info("Testing enhanced autocomplete suggestions...")
    
    test_queries = [
        "iphone", "samsung galaxy", "laptop", "running shoes", 
        "wireless headphones", "smart watch", "gaming", "home"
    ]
    
    from core.ecommerce_autocomplete import ECommerceAutocompleteEngine
    
    # Load product data for testing
    try:
        processed_data_dir = Path("processed_data")
        if (processed_data_dir / "products.parquet").exists():
            import pandas as pd
            product_df = pd.read_parquet(processed_data_dir / "products.parquet")
            product_data = product_df.to_dict('records')
        else:
            product_data = []
    except:
        product_data = []
    
    # Initialize autocomplete engine
    autocomplete_engine = ECommerceAutocompleteEngine(
        cadence_model=model,
        vocab=vocab,
        product_data=product_data
    )
    
    for query in test_queries:
        try:
            suggestions = await autocomplete_engine.get_suggestions(query, max_suggestions=5)
            logger.info(f"Query: '{query}' -> {len(suggestions)} suggestions")
            for i, suggestion in enumerate(suggestions[:3]):
                logger.info(f"  {i+1}. {suggestion['text']} (type: {suggestion['type']}, score: {suggestion['score']:.2f})")
        except Exception as e:
            logger.error(f"Error testing query '{query}': {e}")

def main():
    """Main entry point"""
    # Check if Gemini API key is available
    if not settings.GEMINI_API_KEY and not os.environ.get("GEMINI_API_KEY"):
        logger.error("❌ GEMINI_API_KEY not found in settings or environment variables")
        logger.info("Please set the API key in config/settings.py or as an environment variable")
        return False
    
    logger.info(f"Using Gemini API key: {settings.GEMINI_API_KEY[:10]}...")
    
    # Run the setup
    success = asyncio.run(setup_enhanced_cadence_system())
    
    if success:
        logger.info("🚀 Enhanced CADENCE System is ready!")
        logger.info("You can now run the API server with: python api/main.py")
    else:
        logger.error("❌ Setup failed. Please check the logs above.")
    
    return success

if __name__ == "__main__":
    # Configure logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    main()