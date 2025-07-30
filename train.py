#!/usr/bin/env python3
"""
Training script for CADENCE models.
Run with: python train.py
"""
import logging
from training.train_models import main as train_main

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    train_main()

if __name__ == "__main__":
    main() 