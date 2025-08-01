#!/usr/bin/env python3
"""
🔥 LEGENDARY CADENCE TRAINING SYSTEM 🔥
- Uses the SOPHISTICATED GRU-MN architecture with memory and attention
- Real Amazon QAC + Products data with advanced clustering
- Memory optimized for RTX 3050 4GB but with FULL POWER
- Proper Query LM + Catalog LM training
- Advanced beam search and generation
- Production-ready legendary implementation
"""
import os
import gc
import json
import pickle
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
import numpy as np
import pandas as pd
import structlog
from tqdm import tqdm

# Import the LEGENDARY CADENCE model
from core.cadence_model import CADENCEModel, DynamicBeamSearch

logger = structlog.get_logger()

class LegendaryCADENCEDataset(Dataset):
    """
    Dataset for LEGENDARY CADENCE model training
    Handles both query and catalog training with proper format consistency
    """
    
    def __init__(self, texts: List[str], cluster_ids: List[int], vocab: Dict[str, int], 
                 max_length: int = 64, training_mode: str = 'sequence'):
        self.texts = texts
        self.cluster_ids = cluster_ids
        self.vocab = vocab
        self.max_length = max_length
        self.training_mode = training_mode  # 'sequence' or 'autoregressive'
        
        # Special tokens
        self.pad_token = vocab.get('<PAD>', 0)
        self.unk_token = vocab.get('<UNK>', 1)
        self.eos_token = vocab.get('</s>', 2)
        self.bos_token = vocab.get('<s>', 3)
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        cluster_id = self.cluster_ids[idx]
        
        # Tokenize and convert to IDs
        tokens = text.split()
        token_ids = [self.bos_token]  # Start with BOS
        
        for token in tokens[:self.max_length-2]:  # Reserve space for BOS/EOS
            token_id = self.vocab.get(token, self.unk_token)
            token_ids.append(token_id)
        
        token_ids.append(self.eos_token)  # End with EOS
        
        if self.training_mode == 'sequence':
            # Sequence-to-sequence training (teacher forcing)
            input_ids = token_ids[:-1]  # All except last
            target_ids = token_ids[1:]  # All except first
            
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'target_ids': torch.tensor(target_ids, dtype=torch.long),
                'cluster_id': torch.tensor(cluster_id, dtype=torch.long),
                'length': len(input_ids)
            }
        
        elif self.training_mode == 'autoregressive':
            # Autoregressive training (next token prediction)
            examples = []
            
            for i in range(1, len(token_ids)):
                input_seq = token_ids[:i]
                target_token = token_ids[i]
                
                examples.append({
                    'input_ids': torch.tensor(input_seq, dtype=torch.long),
                    'target_ids': torch.tensor([target_token], dtype=torch.long),
                    'cluster_id': torch.tensor(cluster_id, dtype=torch.long),
                    'length': len(input_seq)
                })
            
            # Return random example from this text
            if examples:
                return examples[np.random.randint(len(examples))]
            else:
                # Fallback
                return {
                    'input_ids': torch.tensor([self.bos_token], dtype=torch.long),
                    'target_ids': torch.tensor([self.eos_token], dtype=torch.long),
                    'cluster_id': torch.tensor(cluster_id, dtype=torch.long),
                    'length': 1
                }

def legendary_collate_fn(batch):
    """
    Advanced collate function for LEGENDARY CADENCE training
    Handles variable length sequences with proper padding
    """
    # Sort batch by length for more efficient packing
    batch = sorted(batch, key=lambda x: x['length'], reverse=True)
    
    # Extract components
    input_ids_list = [item['input_ids'] for item in batch]
    target_ids_list = [item['target_ids'] for item in batch]
    cluster_ids = torch.stack([item['cluster_id'] for item in batch])
    lengths = torch.tensor([item['length'] for item in batch])
    
    # Pad sequences
    input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
    target_ids_padded = pad_sequence(target_ids_list, batch_first=True, padding_value=0)
    
    # Create attention mask
    batch_size, max_len = input_ids_padded.shape
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    for i, length in enumerate(lengths):
        attention_mask[i, :length] = True
    
    # Create category tensor (same category for all positions in sequence)
    category_ids = cluster_ids.unsqueeze(1).expand(batch_size, max_len)
    
    return {
        'input_ids': input_ids_padded,
        'target_ids': target_ids_padded,
        'category_ids': category_ids,
        'attention_mask': attention_mask,
        'lengths': lengths
    }

class LegendaryCADENCETrainer:
    """
    Trainer for the LEGENDARY CADENCE model
    Advanced training with proper memory management and optimization
    """
    
    def __init__(self):
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = torch.cuda.is_available()
        
        # Memory-optimized settings for RTX 3050 4GB
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory < 6:  # RTX 3050 4GB
                self.batch_size = 8
                self.gradient_accumulation_steps = 4
            else:
                self.batch_size = 16
                self.gradient_accumulation_steps = 2
        else:
            self.batch_size = 32
            self.gradient_accumulation_steps = 1
            
        self.max_grad_norm = 1.0
        self.warmup_steps = 100
        
        logger.info("🔥 LEGENDARY CADENCE Trainer initialized:")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB" if torch.cuda.is_available() else "   CPU only")
        logger.info(f"   Mixed precision: {self.use_amp}")
        logger.info(f"   Batch size: {self.batch_size}")
        logger.info(f"   Gradient accumulation: {self.gradient_accumulation_steps}")
    
    def load_processed_data(self, data_dir: str = "processed_data"):
        """Load processed data for legendary training"""
        data_path = Path(data_dir)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Processed data not found at {data_path}")
        
        logger.info(f"📂 Loading processed data from {data_path}/...")
        
        # Load dataframes
        query_df = pd.read_parquet(data_path / "queries.parquet")
        product_df = pd.read_parquet(data_path / "products.parquet")
        
        # Load vocabulary
        with open(data_path / "vocabulary.pkl", 'rb') as f:
            vocab = pickle.load(f)
        
        # Load cluster mappings
        with open(data_path / "cluster_mappings.json", 'r') as f:
            cluster_info = json.load(f)
        
        logger.info(f"✅ Loaded processed data:")
        logger.info(f"   Queries: {len(query_df):,}")
        logger.info(f"   Products: {len(product_df):,}")
        logger.info(f"   Vocabulary: {len(vocab):,}")
        logger.info(f"   Query clusters: {cluster_info['total_query_clusters']}")
        logger.info(f"   Product clusters: {cluster_info['total_product_clusters']}")
        
        return query_df, product_df, vocab, cluster_info
    
    def create_legendary_datasets(self, query_df: pd.DataFrame, product_df: pd.DataFrame,
                                vocab: Dict[str, int], test_split: float = 0.15):
        """Create datasets for legendary training"""
        logger.info("📊 Creating LEGENDARY training datasets...")
        
        # Prepare query data
        query_texts = query_df['processed_query'].tolist()
        query_clusters = query_df['cluster_id'].tolist()
        
        # Prepare product data
        product_texts = product_df['processed_title'].tolist()
        product_clusters = product_df['cluster_id'].tolist()
        
        # Add offset to product cluster IDs to avoid collision with query clusters
        max_query_cluster = max(query_clusters) if query_clusters else 0
        product_clusters = [cid + max_query_cluster + 1 for cid in product_clusters]
        
        # Split data
        query_split_idx = int(len(query_texts) * (1 - test_split))
        product_split_idx = int(len(product_texts) * (1 - test_split))
        
        # Create datasets
        datasets = {
            'query_train': LegendaryCADENCEDataset(
                query_texts[:query_split_idx],
                query_clusters[:query_split_idx],
                vocab,
                max_length=64,
                training_mode='sequence'
            ),
            'query_test': LegendaryCADENCEDataset(
                query_texts[query_split_idx:],
                query_clusters[query_split_idx:],
                vocab,
                max_length=64,
                training_mode='sequence'
            ),
            'product_train': LegendaryCADENCEDataset(
                product_texts[:product_split_idx],
                product_clusters[:product_split_idx],
                vocab,
                max_length=64,
                training_mode='sequence'
            ),
            'product_test': LegendaryCADENCEDataset(
                product_texts[product_split_idx:],
                product_clusters[product_split_idx:],
                vocab,
                max_length=64,
                training_mode='sequence'
            )
        }
        
        logger.info(f"✅ Created LEGENDARY datasets:")
        for name, dataset in datasets.items():
            logger.info(f"   {name}: {len(dataset):,} samples")
        
        return datasets
    
    def create_legendary_model(self, vocab_size: int, num_categories: int) -> CADENCEModel:
        """Create the LEGENDARY CADENCE model"""
        logger.info("🚀 Creating LEGENDARY CADENCE Model...")
        
        # LEGENDARY configuration - optimized for RTX 3050 but still powerful
        config = {
            'vocab_size': vocab_size,
            'num_categories': num_categories,
            'embedding_dim': 256,
            'hidden_dims': [512, 384],  # Multi-layer GRU-MN
            'attention_dims': [256, 192],  # Multi-head attention
            'dropout': 0.2,
            'max_memory_length': 128
        }
        
        model = CADENCEModel(config)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"🔥 LEGENDARY CADENCE Model created:")
        logger.info(f"   Vocabulary: {vocab_size:,}")
        logger.info(f"   Categories: {num_categories}")
        logger.info(f"   Embedding dim: {config['embedding_dim']}")
        logger.info(f"   Hidden dims: {config['hidden_dims']}")
        logger.info(f"   Attention dims: {config['attention_dims']}")
        logger.info(f"   Memory length: {config['max_memory_length']}")
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(f"   Est. memory: {(total_params * 4) / 1e6:.1f}MB")
        
        return model
    
    def train_legendary_model(self, model: CADENCEModel, train_dataset: Dataset, 
                            test_dataset: Dataset, model_type: str = 'query',
                            epochs: int = 5, save_path: str = None):
        """Train the LEGENDARY CADENCE model"""
        logger.info(f"🔥 Training LEGENDARY {model_type.upper()} model for {epochs} epochs...")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=legendary_collate_fn,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=torch.cuda.is_available()
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            collate_fn=legendary_collate_fn,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        # Move model to device
        model.to(self.device)
        
        # Optimizer with warmup and weight decay
        optimizer = AdamW(
            model.parameters(), 
            lr=1e-4,  # Lower LR for stability
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        
        # Mixed precision scaler
        scaler = GradScaler() if self.use_amp else None
        
        # Training tracking
        best_loss = float('inf')
        patience = 3
        patience_counter = 0
        
        for epoch in range(epochs):
            logger.info(f"\n🚀 Epoch {epoch+1}/{epochs}")
            
            # Training phase
            model.train()
            total_loss = 0
            num_batches = 0
            
            pbar = tqdm(train_loader, desc=f"Training {model_type}")
            
            for batch_idx, batch in enumerate(pbar):
                # Reset model memory between batches to avoid gradient accumulation issues
                if hasattr(model, 'query_lm') and hasattr(model.query_lm, 'reset_memory'):
                    model.query_lm.reset_memory()
                if hasattr(model, 'catalog_lm') and hasattr(model.catalog_lm, 'reset_memory'):
                    model.catalog_lm.reset_memory()
                
                # Move to device
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                target_ids = batch['target_ids'].to(self.device, non_blocking=True)
                category_ids = batch['category_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                
                # Forward pass with mixed precision
                if self.use_amp:
                    with autocast():
                        outputs = model(input_ids, category_ids, target_ids, model_type=model_type)
                        loss = outputs['loss']
                        loss = loss / self.gradient_accumulation_steps
                else:
                    outputs = model(input_ids, category_ids, target_ids, model_type=model_type)
                    loss = outputs['loss']
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                if self.use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    if self.use_amp:
                        scaler.unscale_(optimizer)
                        clip_grad_norm_(model.parameters(), self.max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        clip_grad_norm_(model.parameters(), self.max_grad_norm)
                        optimizer.step()
                    
                    optimizer.zero_grad()
                    scheduler.step()
                
                # Update metrics
                total_loss += loss.item() * self.gradient_accumulation_steps
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
                
                # Memory cleanup
                if batch_idx % 50 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
            
            avg_train_loss = total_loss / num_batches
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Validation"):
                    # Reset model memory for validation to handle different batch sizes
                    if hasattr(model, 'query_lm') and hasattr(model.query_lm, 'reset_memory'):
                        model.query_lm.reset_memory()
                    if hasattr(model, 'catalog_lm') and hasattr(model.catalog_lm, 'reset_memory'):
                        model.catalog_lm.reset_memory()
                    
                    input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                    target_ids = batch['target_ids'].to(self.device, non_blocking=True)
                    category_ids = batch['category_ids'].to(self.device, non_blocking=True)
                    
                    if self.use_amp:
                        with autocast():
                            outputs = model(input_ids, category_ids, target_ids, model_type=model_type)
                            loss = outputs['loss']
                    else:
                        outputs = model(input_ids, category_ids, target_ids, model_type=model_type)
                        loss = outputs['loss']
                    
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            
            # Logging
            logger.info(f"Epoch {epoch+1} Results:")
            logger.info(f"   Train Loss: {avg_train_loss:.4f}")
            logger.info(f"   Val Loss: {avg_val_loss:.4f}")
            logger.info(f"   Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
            
            # Early stopping and model saving
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                patience_counter = 0
                
                if save_path:
                    logger.info(f"💾 Saving best model to {save_path}")
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'epoch': epoch,
                        'best_loss': best_loss,
                        'config': model.config
                    }, save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                    break
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        logger.info(f"✅ Training completed! Best validation loss: {best_loss:.4f}")
        return model

def main():
    """Main legendary training function"""
    logger.info("🔥" * 40)
    logger.info("🚀 LEGENDARY CADENCE TRAINING SYSTEM 🚀")
    logger.info("🔥" * 40)
    logger.info("Features:")
    logger.info("• Sophisticated GRU-MN architecture with memory")
    logger.info("• Multi-head attention and beam search")
    logger.info("• Real Amazon QAC + clustered categories")
    logger.info("• Advanced training with mixed precision")
    logger.info("• Memory optimized for RTX 3050 4GB")
    logger.info("• Early stopping and learning rate scheduling")
    logger.info("")
    
    try:
        # Initialize trainer
        trainer = LegendaryCADENCETrainer()
        
        # Load processed data
        logger.info("STEP 1/5: Loading Processed Data")
        logger.info("-" * 50)
        query_df, product_df, vocab, cluster_info = trainer.load_processed_data()
        
        # Create legendary datasets
        logger.info("\nSTEP 2/5: Creating LEGENDARY Datasets")
        logger.info("-" * 50)
        datasets = trainer.create_legendary_datasets(query_df, product_df, vocab)
        
        # Create legendary model
        logger.info("\nSTEP 3/5: Creating LEGENDARY Model")
        logger.info("-" * 50)
        total_categories = (cluster_info['total_query_clusters'] + 
                          cluster_info['total_product_clusters'])
        model = trainer.create_legendary_model(len(vocab), total_categories)
        
        # Train Query LM
        logger.info("\nSTEP 4/5: Training LEGENDARY Query Model")
        logger.info("-" * 50)
        Path("legendary_models").mkdir(exist_ok=True)
        query_model_path = "legendary_models/legendary_query_model.pt"
        
        model = trainer.train_legendary_model(
            model,
            datasets['query_train'],
            datasets['query_test'],
            model_type='query',
            epochs=5,
            save_path=query_model_path
        )
        
        # Train Catalog LM
        logger.info("\nSTEP 5/5: Training LEGENDARY Catalog Model")
        logger.info("-" * 50)
        catalog_model_path = "legendary_models/legendary_catalog_model.pt"
        
        model = trainer.train_legendary_model(
            model,
            datasets['product_train'],
            datasets['product_test'],
            model_type='catalog',
            epochs=5,
            save_path=catalog_model_path
        )
        
        # Save complete model
        logger.info("\n💾 Saving Complete LEGENDARY Model")
        logger.info("-" * 50)
        
        complete_model_data = {
            'model_state_dict': model.state_dict(),
            'model_config': model.config,
            'vocab': vocab,
            'cluster_info': cluster_info,
            'training_date': datetime.now().isoformat()
        }
        
        complete_model_path = "legendary_models/legendary_cadence_complete.pt"
        torch.save(complete_model_data, complete_model_path)
        
        # Create beam search
        beam_search = DynamicBeamSearch(model, vocab, beam_width=10, max_length=50)
        
        logger.info("\n🔥" * 40)
        logger.info("✅ LEGENDARY CADENCE TRAINING COMPLETED!")
        logger.info("🔥" * 40)
        logger.info(f"📊 Results:")
        logger.info(f"   Model architecture: GRU-MN with memory & attention")
        logger.info(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"   Query model: {query_model_path}")
        logger.info(f"   Catalog model: {catalog_model_path}")
        logger.info(f"   Complete model: {complete_model_path}")
        logger.info("")
        logger.info("🚀 Ready for LEGENDARY inference and beam search!")
        
    except Exception as e:
        logger.error(f"❌ LEGENDARY training failed: {e}")
        logger.exception("Full error traceback:")
        raise

if __name__ == "__main__":
    main()