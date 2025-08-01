#!/usr/bin/env python3
"""
SUPER-OPTIMIZED Training Script for CADENCE Models
Designed for RTX 3050 4GB + 16GB RAM + Ryzen 7 6800H

Features:
- Memory-optimized model architecture 
- Mixed precision training (FP16)
- Aggressive GPU memory management
- Checkpointing with resume capability
- Background training support
- Multi-core CPU utilization
- Streaming data processing
- Real-time memory monitoring
"""
import os
import gc
import sys
import time
import json
import pickle
import psutil
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
from datasets import load_dataset
import structlog
from tqdm import tqdm

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
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger()

class MemoryMonitor:
    """Real-time memory monitoring and management"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.monitoring = False
        self.thread = None
        
    def start_monitoring(self):
        """Start memory monitoring in background thread"""
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=1)
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            ram_mb = self.process.memory_info().rss / 1024 / 1024
            
            if torch.cuda.is_available():
                gpu_allocated = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_cached = torch.cuda.memory_reserved() / 1024 / 1024
                
                # Critical memory management
                if gpu_allocated > 3500:  # 3.5GB threshold for 4GB GPU
                    logger.warning(f"GPU memory critical: {gpu_allocated:.1f}MB allocated")
                    self.emergency_cleanup()
                    
                if ram_mb > 14000:  # 14GB threshold for 16GB RAM
                    logger.warning(f"RAM critical: {ram_mb:.1f}MB used")
                    self.emergency_cleanup()
                    
                logger.debug(f"Memory: RAM {ram_mb:.1f}MB, GPU {gpu_allocated:.1f}MB allocated, {gpu_cached:.1f}MB cached")
            else:
                logger.debug(f"Memory: RAM {ram_mb:.1f}MB")
                
            time.sleep(10)  # Check every 10 seconds
    
    def emergency_cleanup(self):
        """Emergency memory cleanup"""
        logger.info("Performing emergency memory cleanup...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
    def log_memory_usage(self, step_name: str):
        """Log current memory usage"""
        ram_mb = self.process.memory_info().rss / 1024 / 1024
        
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / 1024 / 1024
            gpu_cached = torch.cuda.memory_reserved() / 1024 / 1024
            logger.info(f"{step_name} - RAM: {ram_mb:.1f}MB, GPU: {gpu_allocated:.1f}MB allocated, {gpu_cached:.1f}MB cached")
        else:
            logger.info(f"{step_name} - RAM: {ram_mb:.1f}MB")

class OptimizedCADENCEModel(nn.Module):
    """Memory-optimized CADENCE model for RTX 3050 4GB"""
    
    def __init__(self, vocab_size: int, num_categories: int):
        super().__init__()
        
        # Optimized dimensions for 4GB GPU
        self.vocab_size = vocab_size
        self.num_categories = num_categories
        self.embedding_dim = 128  # Reduced from 256
        self.hidden_dim = 256     # Reduced from 2000/1500/1000
        self.dropout = 0.3        # Reduced from 0.8
        
        # Embeddings with reduced dimensions
        self.word_embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.category_embedding = nn.Embedding(num_categories, self.embedding_dim // 2)
        
        # Single GRU layer instead of 3-layer architecture
        self.gru = nn.GRU(
            input_size=self.embedding_dim + self.embedding_dim // 2,
            hidden_size=self.hidden_dim,
            num_layers=1,  # Single layer for memory efficiency
            batch_first=True,
            dropout=0
        )
        
        # Simplified output layers
        self.dropout_layer = nn.Dropout(self.dropout)
        self.output_projection = nn.Linear(self.hidden_dim, vocab_size)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        
        logger.info(f"Created optimized CADENCE model:")
        logger.info(f"- Vocab size: {vocab_size}")
        logger.info(f"- Categories: {num_categories}")  
        logger.info(f"- Embedding dim: {self.embedding_dim}")
        logger.info(f"- Hidden dim: {self.hidden_dim}")
        logger.info(f"- Total parameters: ~{self.count_parameters()/1e6:.1f}M")
        
    def count_parameters(self):
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def forward(self, input_ids: torch.Tensor, category_ids: torch.Tensor, 
                target_ids: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Forward pass with memory optimization"""
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        word_embeds = self.word_embedding(input_ids)
        category_embeds = self.category_embedding(category_ids)
        
        # Concatenate embeddings
        input_embeds = torch.cat([word_embeds, category_embeds], dim=-1)
        
        # GRU forward
        gru_output, _ = self.gru(input_embeds)
        
        # Apply dropout
        gru_output = self.dropout_layer(gru_output)
        
        # Output projection
        logits = self.output_projection(gru_output)
        
        result = {'logits': logits}
        
        # Calculate loss if targets provided
        if target_ids is not None:
            loss = self.criterion(logits.view(-1, self.vocab_size), target_ids.view(-1))
            result['loss'] = loss
            
        return result

class StreamingDataset(Dataset):
    """Ultra-efficient streaming dataset"""
    
    def __init__(self, texts: List[str], cluster_ids: List[int], vocab: Dict[str, int], max_length: int = 32):
        self.texts = texts
        self.cluster_ids = cluster_ids
        self.vocab = vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        cluster_id = self.cluster_ids[idx]
        
        # Tokenize with truncation
        tokens = text.split()[:self.max_length-2]
        token_ids = [self.vocab.get('<s>', 3)]  # BOS
        
        for token in tokens:
            token_ids.append(self.vocab.get(token, self.vocab.get('<UNK>', 1)))
        
        token_ids.append(self.vocab.get('</s>', 2))  # EOS
        
        # Create input/target sequences
        input_ids = token_ids[:-1]
        target_ids = token_ids[1:]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'cluster_id': torch.tensor(cluster_id, dtype=torch.long)
        }

def optimized_collate_fn(batch):
    """Memory-optimized collate function"""
    input_ids = [item['input_ids'] for item in batch]
    target_ids = [item['target_ids'] for item in batch]
    cluster_ids = torch.stack([item['cluster_id'] for item in batch])
    
    # Pad sequences
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    target_ids_padded = pad_sequence(target_ids, batch_first=True, padding_value=0)
    
    # Create category tensor
    batch_size, seq_len = input_ids_padded.shape
    category_ids = cluster_ids.unsqueeze(1).expand(batch_size, seq_len)
    
    return {
        'input_ids': input_ids_padded,
        'target_ids': target_ids_padded,
        'category_ids': category_ids
    }

class OptimizedCADENCETrainer:
    """Super-optimized trainer for RTX 3050 4GB"""
    
    def __init__(self, model_dir: str = "models", checkpoint_dir: str = "checkpoints"):
        self.model_dir = Path(model_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Setup device and optimizations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_cuda = torch.cuda.is_available()
        self.use_amp = self.use_cuda  # Use mixed precision if CUDA available
        
        # Memory monitoring
        self.memory_monitor = MemoryMonitor()
        
        # Optimization settings for RTX 3050 4GB
        self.batch_size = 8 if self.use_cuda else 16  # Smaller batch for GPU
        self.gradient_accumulation_steps = 4  # Simulate larger batch
        self.max_grad_norm = 1.0
        
        logger.info(f"Initialized optimized trainer:")
        logger.info(f"- Device: {self.device}")
        logger.info(f"- Mixed precision: {self.use_amp}")
        logger.info(f"- Batch size: {self.batch_size}")
        logger.info(f"- Gradient accumulation: {self.gradient_accumulation_steps}")
        logger.info(f"- Effective batch size: {self.batch_size * self.gradient_accumulation_steps}")
        
    def create_tiny_dataset(self, num_samples: int = 5000) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Create tiny dataset for memory-efficient training"""
        logger.info(f"Creating tiny dataset with {num_samples} samples...")
        
        # Sample queries and products
        sample_queries = [
            "laptop gaming computer", "wireless bluetooth headphones", "running shoes men size",
            "smartphone android unlocked", "coffee maker automatic brew", "book fiction bestseller",
            "tablet screen protector glass", "kitchen knife set steel", "winter jacket waterproof",
            "fitness tracker smartwatch", "camera digital photography", "desk chair office ergonomic",
            "bluetooth speaker portable waterproof", "phone case protective clear", "laptop bag leather messenger"
        ] * (num_samples // 15 + 1)
        
        sample_products = [
            "Apple MacBook Pro 16 inch M3 Laptop Gaming",
            "Sony WH-1000XM5 Wireless Noise Canceling Headphones",
            "Nike Air Max 270 Running Shoes Men Size 10",
            "Samsung Galaxy S24 Smartphone Android Unlocked 128GB",
            "Breville Bambino Plus Coffee Maker Automatic Espresso",
            "The Seven Husbands Evelyn Hugo Fiction Bestseller Novel",
            "iPad Pro 12.9 inch Tablet Screen Protector Tempered Glass",
            "Wusthof Classic 8 piece Kitchen Knife Set German Steel",
            "Patagonia Down Sweater Winter Jacket Waterproof Men",
            "Fitbit Charge 5 Fitness Tracker Smartwatch Health GPS"
        ] * (num_samples // 10 + 1)
        
        # Create training data
        query_data = []
        for i in range(num_samples):
            query_data.append({
                'text': sample_queries[i % len(sample_queries)],
                'cluster_id': i % 5  # 5 clusters
            })
            
        catalog_data = []
        for i in range(num_samples // 2):  # Fewer catalog samples
            catalog_data.append({
                'text': sample_products[i % len(sample_products)],
                'cluster_id': i % 5  # 5 clusters
            })
        
        # Build vocabulary
        all_texts = [item['text'] for item in query_data + catalog_data]
        vocab = self._build_vocabulary(all_texts)
        
        # Split data
        train_split = int(0.8 * len(query_data))
        query_train = query_data[:train_split]
        query_val = query_data[train_split:]
        
        catalog_train_split = int(0.8 * len(catalog_data))
        catalog_train = catalog_data[:catalog_train_split]
        catalog_val = catalog_data[catalog_train_split:]
        
        query_datasets = {
            'train': query_train,
            'val': query_val,
            'vocab': vocab,
            'clusters': {i: f"cluster_{i}" for i in range(5)}
        }
        
        catalog_datasets = {
            'train': catalog_train,
            'val': catalog_val,
            'vocab': vocab,
            'clusters': {i: f"cluster_{i}" for i in range(5)}
        }
        
        logger.info(f"Created tiny dataset:")
        logger.info(f"- Query train: {len(query_train)} samples")
        logger.info(f"- Query val: {len(query_val)} samples") 
        logger.info(f"- Catalog train: {len(catalog_train)} samples")
        logger.info(f"- Catalog val: {len(catalog_val)} samples")
        logger.info(f"- Vocabulary size: {len(vocab)}")
        
        return query_datasets, catalog_datasets
    
    def _build_vocabulary(self, texts: List[str], max_vocab_size: int = 10000) -> Dict[str, int]:
        """Build vocabulary from texts"""
        from collections import Counter
        
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        
        # Get most common words
        most_common = word_counts.most_common(max_vocab_size - 4)
        
        vocab = {
            '<PAD>': 0,
            '<UNK>': 1,
            '</s>': 2,
            '<s>': 3
        }
        
        for i, (word, _) in enumerate(most_common):
            vocab[word] = i + 4
        
        return vocab
    
    def save_checkpoint(self, model: OptimizedCADENCEModel, optimizer: torch.optim.Optimizer,
                       scaler: GradScaler, epoch: int, step: int, loss: float,
                       vocab: Dict[str, int], config: Dict[str, Any]):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'loss': loss,
            'vocab': vocab,
            'config': config,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Keep only last 3 checkpoints to save disk space
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if len(checkpoints) > 3:
            for old_checkpoint in checkpoints[:-3]:
                old_checkpoint.unlink()
                
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """Load training checkpoint"""
        if not Path(checkpoint_path).exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return None
            
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}, step {checkpoint['step']}")
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint"""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            return None
        
        # Sort by modification time
        latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return str(latest_checkpoint)
    
    def cleanup_memory(self):
        """Aggressive memory cleanup"""
        gc.collect()
        if self.use_cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def train_model_optimized(self, train_data: List[Dict], val_data: List[Dict],
                            vocab: Dict[str, int], epochs: int = 5,
                            resume_from_checkpoint: bool = True) -> OptimizedCADENCEModel:
        """Optimized training with checkpointing and memory management"""
        
        # Start memory monitoring
        self.memory_monitor.start_monitoring()
        self.memory_monitor.log_memory_usage("Training start")
        
        try:
            # Create model
            num_categories = max([item.get('cluster_id', 0) for item in train_data + val_data]) + 1
            model = OptimizedCADENCEModel(len(vocab), num_categories)
            model.to(self.device)
            
            # Setup optimizer and scheduler
            optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            scaler = GradScaler() if self.use_amp else None
            
            # Try to resume from checkpoint
            start_epoch = 0
            start_step = 0
            
            if resume_from_checkpoint:
                latest_checkpoint = self.find_latest_checkpoint()
                if latest_checkpoint:
                    checkpoint = self.load_checkpoint(latest_checkpoint)
                    if checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        if scaler and checkpoint['scaler_state_dict']:
                            scaler.load_state_dict(checkpoint['scaler_state_dict'])
                        start_epoch = checkpoint['epoch']
                        start_step = checkpoint['step']
                        logger.info(f"Resumed from epoch {start_epoch}, step {start_step}")
            
            # Create datasets
            train_texts = [item['text'] for item in train_data]
            train_clusters = [item.get('cluster_id', 0) for item in train_data]
            val_texts = [item['text'] for item in val_data]
            val_clusters = [item.get('cluster_id', 0) for item in val_data]
            
            train_dataset = StreamingDataset(train_texts, train_clusters, vocab)
            val_dataset = StreamingDataset(val_texts, val_clusters, vocab)
            
            # Create data loaders with memory optimization
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=optimized_collate_fn,
                num_workers=2,  # Use 2 CPU cores for data loading
                pin_memory=self.use_cuda,
                persistent_workers=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                collate_fn=optimized_collate_fn,
                num_workers=1,
                pin_memory=self.use_cuda
            )
            
            self.memory_monitor.log_memory_usage("Data loaders created")
            
            # Training loop
            global_step = start_step
            
            for epoch in range(start_epoch, epochs):
                model.train()
                epoch_loss = 0
                num_batches = 0
                
                # Progress bar
                pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
                
                for batch_idx, batch in enumerate(pbar):
                    # Move to device
                    input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                    target_ids = batch['target_ids'].to(self.device, non_blocking=True)
                    category_ids = batch['category_ids'].to(self.device, non_blocking=True)
                    
                    # Forward pass with mixed precision
                    if self.use_amp:
                        with autocast():
                            outputs = model(input_ids, category_ids, target_ids)
                            loss = outputs['loss']
                            # Scale loss for gradient accumulation
                            loss = loss / self.gradient_accumulation_steps
                    else:
                        outputs = model(input_ids, category_ids, target_ids)
                        loss = outputs['loss'] / self.gradient_accumulation_steps
                    
                    # Backward pass
                    if self.use_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    
                    # Gradient accumulation
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        if self.use_amp:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                            optimizer.step()
                        
                        optimizer.zero_grad()
                        global_step += 1
                        
                        # Save checkpoint every 50 steps
                        if global_step % 50 == 0:
                            self.save_checkpoint(
                                model, optimizer, scaler, epoch, global_step,
                                loss.item() * self.gradient_accumulation_steps,
                                vocab, {'num_categories': num_categories}
                            )
                    
                    epoch_loss += loss.item() * self.gradient_accumulation_steps
                    num_batches += 1
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                        'step': global_step
                    })
                    
                    # Memory cleanup every 10 batches
                    if batch_idx % 10 == 0:
                        self.cleanup_memory()
                
                avg_train_loss = epoch_loss / num_batches
                
                # Validation
                model.eval()
                val_loss = 0
                val_batches = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                        target_ids = batch['target_ids'].to(self.device, non_blocking=True)
                        category_ids = batch['category_ids'].to(self.device, non_blocking=True)
                        
                        if self.use_amp:
                            with autocast():
                                outputs = model(input_ids, category_ids, target_ids)
                                loss = outputs['loss']
                        else:
                            outputs = model(input_ids, category_ids, target_ids)
                            loss = outputs['loss']
                        
                        val_loss += loss.item()
                        val_batches += 1
                
                avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
                
                logger.info(f"Epoch {epoch+1}/{epochs} completed:")
                logger.info(f"- Train Loss: {avg_train_loss:.4f}")
                logger.info(f"- Val Loss: {avg_val_loss:.4f}")
                logger.info(f"- Global Step: {global_step}")
                
                self.memory_monitor.log_memory_usage(f"Epoch {epoch+1} complete")
                
                # Save epoch checkpoint
                self.save_checkpoint(
                    model, optimizer, scaler, epoch + 1, global_step,
                    avg_val_loss, vocab, {'num_categories': num_categories}
                )
                
                # Cleanup memory after each epoch
                self.cleanup_memory()
            
            return model
            
        finally:
            # Stop memory monitoring
            self.memory_monitor.stop_monitoring()
    
    def save_final_model(self, model: OptimizedCADENCEModel, vocab: Dict[str, int],
                        cluster_mappings: Dict[str, Any], model_name: str = "optimized_cadence"):
        """Save final trained model"""
        model_path = self.model_dir / f"{model_name}.pt"
        vocab_path = self.model_dir / f"{model_name}_vocab.pkl"
        config_path = self.model_dir / f"{model_name}_config.json"
        
        # Save model state dict
        torch.save(model.state_dict(), model_path)
        
        # Save vocabulary
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f)
        
        # Save configuration
        config = {
            'vocab_size': len(vocab),
            'num_categories': model.num_categories,
            'model_config': {
                'embedding_dim': model.embedding_dim,
                'hidden_dim': model.hidden_dim,
                'dropout': model.dropout
            },
            'cluster_mappings': cluster_mappings,
            'training_date': datetime.now().isoformat(),
            'optimized_for': 'RTX_3050_4GB'
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Final model saved:")
        logger.info(f"- Model: {model_path}")
        logger.info(f"- Vocabulary: {vocab_path}")
        logger.info(f"- Configuration: {config_path}")

def run_optimized_training(epochs: int = 5, tiny_dataset_size: int = 5000,
                          resume_from_checkpoint: bool = True):
    """Run the optimized training pipeline"""
    logger.info("=" * 60)
    logger.info("STARTING SUPER-OPTIMIZED CADENCE TRAINING")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    logger.info(f"- Epochs: {epochs}")
    logger.info(f"- Dataset size: {tiny_dataset_size}")
    logger.info(f"- Resume from checkpoint: {resume_from_checkpoint}")
    logger.info(f"- CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"- GPU: {torch.cuda.get_device_name()}")
        logger.info(f"- GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # Initialize trainer
    trainer = OptimizedCADENCETrainer()
    
    try:
        # Create tiny dataset for testing
        logger.info("Step 1/3: Creating optimized dataset...")
        query_datasets, catalog_datasets = trainer.create_tiny_dataset(tiny_dataset_size)
        
        # Train the model
        logger.info("Step 2/3: Training optimized model...")
        model = trainer.train_model_optimized(
            query_datasets['train'],
            query_datasets['val'],
            query_datasets['vocab'],
            epochs=epochs,
            resume_from_checkpoint=resume_from_checkpoint
        )
        
        # Save final model
        logger.info("Step 3/3: Saving final model...")
        cluster_mappings = {
            'query_clusters': query_datasets['clusters'],
            'catalog_clusters': catalog_datasets['clusters']
        }
        
        trainer.save_final_model(model, query_datasets['vocab'], cluster_mappings)
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        
        return model, query_datasets['vocab'], cluster_mappings
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.exception("Full error traceback:")
        raise

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Super-Optimized CADENCE Training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--dataset-size", type=int, default=5000, help="Size of tiny dataset")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from checkpoint")
    parser.add_argument("--background", action="store_true", help="Run in background")
    
    args = parser.parse_args()
    
    if args.background:
        logger.info("Starting training in background...")
        # Run in background using subprocess
        script_path = Path(__file__).absolute()
        cmd = [
            sys.executable, str(script_path),
            "--epochs", str(args.epochs),
            "--dataset-size", str(args.dataset_size)
        ]
        if args.no_resume:
            cmd.append("--no-resume")
        
        # Start background process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        
        logger.info(f"Background training started with PID: {process.pid}")
        logger.info("Check logs in real-time or use task manager to monitor progress")
        logger.info("The training will continue even if you close this terminal")
        
        return process
    else:
        # Run training directly
        run_optimized_training(
            epochs=args.epochs,
            tiny_dataset_size=args.dataset_size,
            resume_from_checkpoint=not args.no_resume
        )

if __name__ == "__main__":
    main()