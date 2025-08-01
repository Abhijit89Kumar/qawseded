#!/usr/bin/env python3
"""
🚨 DEPRECATED: Original training script with memory issues
Use optimized_train.py instead for better performance and stability
"""
import logging
import sys

def main():
    print("🚨 WARNING: This training script has been DEPRECATED")
    print("")
    print("❌ ISSUES WITH THIS SCRIPT:")
    print("   • Crashes on RTX 3050 4GB GPU")
    print("   • Uses too much RAM (16GB+)")
    print("   • No memory optimization")
    print("   • No checkpointing")
    print("   • No background training")
    print("")
    print("✅ USE INSTEAD:")
    print("   • Windows: Double-click start_training.bat")
    print("   • Command line: python optimized_train.py")
    print("   • Background: python run_optimized_training.py start")
    print("")
    print("🔧 OPTIMIZED FEATURES:")
    print("   • Memory-optimized for RTX 3050 4GB")
    print("   • Mixed precision training (FP16)")
    print("   • Aggressive memory cleanup")
    print("   • Checkpointing with resume")
    print("   • Background training support")
    print("   • Real-time monitoring")
    print("")
    
    choice = input("Continue with deprecated script anyway? (y/N): ")
    if choice.lower() != 'y':
        print("Smart choice! Use the optimized training instead. 👍")
        sys.exit(0)
    
    print("⚠️  Proceeding with deprecated script at your own risk...")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    try:
        from training.train_models import main as train_main
        train_main()
    except Exception as e:
        print(f"\n💥 TRAINING FAILED: {e}")
        print("\nThis is why we created the optimized version!")
        print("Please use: python optimized_train.py")
        sys.exit(1)

if __name__ == "__main__":
    main() 