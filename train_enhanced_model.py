#!/usr/bin/env python3
"""
Train the enhanced CADENCE model
"""
from training.train_models import CADENCETrainer
import structlog
import sys

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

def main():
    logger.info("🚀 Starting enhanced CADENCE model training...")
    
    try:
        # Initialize trainer
        trainer = CADENCETrainer()
        logger.info("✅ Trainer initialized")
        
        # Prepare data with smaller sample size for faster processing
        logger.info("📊 Preparing training data (using 50K samples for faster processing)...")
        training_data = trainer.prepare_data(max_samples=50000)  # Reduced from 1M to 50K
        logger.info("✅ Training data prepared")
        
        # Train enhanced model
        logger.info("🧠 Training enhanced CADENCE model (3 epochs)...")
        result = trainer.train_enhanced_models(
            training_data, 
            epochs=3, 
            save_name='enhanced_cadence_production'
        )
        
        logger.info("✅ Enhanced model training completed successfully!")
        logger.info("🎉 Model saved as 'enhanced_cadence_production'")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        logger.error("Please check the error details above and try again")
        sys.exit(1)

if __name__ == "__main__":
    main()