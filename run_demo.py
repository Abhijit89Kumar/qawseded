#!/usr/bin/env python3
"""
Enhanced CADENCE System Demo Runner

This script sets up and runs the complete Enhanced CADENCE demo including:
1. Environment setup
2. Database initialization  
3. Model training (if needed)
4. Web server startup
5. Demo instructions

Run with: python run_demo.py
"""

import os
import sys
import asyncio
import subprocess
import logging
from pathlib import Path
import uvicorn
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup environment variables for demo"""
    env_vars = {
        'DATABASE_URL': 'sqlite:///./cadence.db',
        'REDIS_URL': 'redis://localhost:6379',
        'GEMINI_API_KEY': 'demo-key-not-needed-for-local',
        'EMBEDDING_MODEL': 'sentence-transformers/all-MiniLM-L6-v2',
        'EMBEDDING_DIM': '384',
        'VOCAB_SIZE': '10000',
        'HIDDEN_DIMS': '[512,256,128]',
        'ATTENTION_DIMS': '[256,128,64]',
        'DROPOUT_RATE': '0.3',
        'LEARNING_RATE': '0.001',
        'BATCH_SIZE': '32',
        'N_CLUSTERS': '10',
        'MIN_CLUSTER_SIZE': '5',
        'MIN_SAMPLES': '3',
        'MAX_USER_HISTORY': '500',
        'MAX_SUGGESTIONS': '10',
        'CACHE_TTL': '300',
        'ENGAGEMENT_WEIGHT_VIEW': '1.0',
        'ENGAGEMENT_WEIGHT_CLICK': '2.0',
        'ENGAGEMENT_WEIGHT_ADD_TO_CART': '5.0',
        'ENGAGEMENT_WEIGHT_WISHLIST': '3.0',
        'ENGAGEMENT_WEIGHT_PURCHASE': '10.0',
        'MAX_LATENCY_MS': '200',
        'AUTOSUGGEST_TIMEOUT_MS': '100',
        'SEARCH_TIMEOUT_MS': '300',
        'API_HOST': '0.0.0.0',
        'API_PORT': '8000',
        'LOG_LEVEL': 'INFO',
        'DEBUG': 'true',
        'DEVELOPMENT_MODE': 'true',
        'ENABLE_AB_TESTING': 'true',
        'ENABLE_CLUSTERING_CATEGORIES': 'true',
        'ENABLE_REAL_TIME_UPDATES': 'true'
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    logger.info("✅ Environment variables configured")

def check_dependencies():
    """Check if required dependencies are installed"""
    logger.info("Checking dependencies...")
    
    try:
        import torch
        import numpy
        import pandas
        import sklearn
        import fastapi
        import uvicorn
        import structlog
        import sqlalchemy
        logger.info("✅ Core dependencies found")
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.info("Please install requirements: pip install -r requirements.txt")
        sys.exit(1)
    
    # Check optional dependencies
    try:
        import redis
        logger.info("✅ Redis dependency found")
    except ImportError:
        logger.warning("⚠️ Redis not available - caching will be disabled")
    
    try:
        import datasets
        logger.info("✅ Datasets library found")
    except ImportError:
        logger.warning("⚠️ Datasets library not found - will use synthetic data")

def create_directories():
    """Create necessary directories"""
    directories = [
        'models',
        'static',
        'logs',
        'data'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    logger.info("✅ Directory structure created")

async def initialize_system():
    """Initialize the system components"""
    logger.info("Initializing Enhanced CADENCE system...")
    
    # Import here to avoid circular imports
    from database.connection import initialize_database
    
    try:
        # Initialize database
        success = await initialize_database()
        if success:
            logger.info("✅ Database initialized")
        else:
            logger.error("❌ Database initialization failed")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        return False

def train_models_if_needed():
    """Ensure pre-trained models exist and instruct user to train if missing"""
    models_dir = Path('models')
    model_files = [
        'cadence_trained.pt',
        'cadence_trained_vocab.pkl',
        'cadence_trained_config.json'
    ]
    
    models_exist = all((models_dir / file).exists() for file in model_files)
    
    if not models_exist:
        logger.error("❌ Pre-trained models not found. Please run the training script: python train.py")
        sys.exit(1)
    logger.info("✅ Pre-trained models found")
    return True

def print_demo_banner():
    """Print attractive demo banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                          🚀 Enhanced CADENCE Demo                            ║
║                    Hyper-Personalized E-commerce Search                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Features Demonstrated:                                                      ║
║  ✨ Real-time Query Autocompletion                                          ║
║  🎯 Personalized Search Results                                             ║
║  🧠 Machine Learning-powered Suggestions                                    ║
║  📊 User Behavior Tracking & Analytics                                      ║
║  🔄 Dynamic Personalization Engine                                          ║
║                                                                              ║
║  Architecture Highlights:                                                   ║
║  🏗️  SQLite Database (Local Development)                                   ║
║  🤖 PyTorch-based CADENCE Models                                           ║
║  ⚡ FastAPI Backend with Real-time Updates                                  ║
║  🎨 Modern Responsive Web Interface                                         ║
║  📈 A/B Testing Framework                                                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_demo_instructions():
    """Print instructions for using the demo"""
    instructions = """
🎯 Demo Instructions:
═══════════════════

1. 🌐 Open your browser and go to: http://localhost:8000

2. 🔍 Try searching for:
   • "laptop" - See gaming, programming, business suggestions
   • "headphones" - See wireless, bluetooth, noise-cancelling options  
   • "shoes" - See running, casual, sports recommendations
   • "phone" - See case, charger, accessory suggestions

3. 👤 User Personalization:
   • Change the User ID to see different personalized results
   • Use the engagement buttons to simulate user interactions
   • Watch how suggestions improve over time

4. ⚙️ Advanced Features:
   • Toggle personalization on/off to see the difference
   • Adjust max suggestions count
   • Monitor real-time statistics

5. 📊 Analytics:
   • View total queries, response times, personalization rates
   • See how engagement affects future recommendations
   • Experience the power of hyper-personalization!

═══════════════════════════════════════════════════════════════════════════════

📖 Technical Details:
• Database: SQLite with real-time caching
• Models: PyTorch-based CADENCE architecture  
• Frontend: Vanilla JavaScript with modern CSS
• API: FastAPI with async/await patterns

💡 Tips:
• Try different search patterns to see clustering in action
• Use multiple browser tabs with different User IDs
• Simulate realistic shopping behavior for best results

Press Ctrl+C to stop the demo server
═══════════════════════════════════════════════════════════════════════════════
    """
    print(instructions)

async def run_demo():
    """Main demo runner"""
    print_demo_banner()
    
    logger.info("🚀 Starting Enhanced CADENCE Demo...")
    
    # Setup
    setup_environment()
    check_dependencies()
    create_directories()
    
    # Initialize system
    initialized = await initialize_system()
    if not initialized:
        logger.error("❌ System initialization failed - some features may not work")
    
    # Train models if needed
    train_models_if_needed()
    
    print_demo_instructions()
    
    # Start the web server
    logger.info("🌐 Starting web server on http://localhost:8000")
    
    try:
        # Import the FastAPI app
        from api.main import app
        
        # Run the server
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            reload=False,  # Disable reload for demo
            access_log=True
        )
        
        server = uvicorn.Server(config)
        await server.serve()
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Demo stopped by user")
        print("\n✨ Thanks for trying Enhanced CADENCE!")
        print("For more information, check out the README.md file\n")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        sys.exit(1)

def main():
    """Entry point for the demo"""
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        logger.info("\n🛑 Demo interrupted")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 