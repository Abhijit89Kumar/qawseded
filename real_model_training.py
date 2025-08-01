#!/usr/bin/env python3
"""
REAL CADENCE Model Training
Trains both Query LM and Catalog LM using processed Amazon data
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
import numpy as np
import pandas as pd
import structlog
from tqdm import tqdm

from real_cadence_training import RealCADENCEModel

logger = structlog.get_logger()

class CADENCEDataset(Dataset):
    """Dataset for CADENCE model training"""
    
    def __init__(self, texts: List[str], cluster_ids: List[int], vocab: Dict[str, int], 
                 max_length: int = 64, is_autocomplete: bool = False):
        self.texts = texts
        self.cluster_ids = cluster_ids
        self.vocab = vocab
        self.max_length = max_length
        self.is_autocomplete = is_autocomplete
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        cluster_id = self.cluster_ids[idx]
        
        # Tokenize text
        tokens = text.split()
        
        if self.is_autocomplete:
            # For autocomplete, create multiple training examples from prefixes
            examples = []
            for i in range(1, min(len(tokens), self.max_length - 1)):
                prefix = tokens[:i]
                target = tokens[i] if i < len(tokens) else '</s>'
                
                # Convert to IDs
                input_ids = [self.vocab.get('<s>', 3)]
                input_ids.extend([self.vocab.get(token, self.vocab.get('<UNK>', 1)) for token in prefix])
                
                target_id = self.vocab.get(target, self.vocab.get('<UNK>', 1))
                if target == '</s>':
                    target_id = self.vocab.get('</s>', 2)
                
                examples.append({
                    'input_ids': torch.tensor(input_ids, dtype=torch.long),
                    'target_id': torch.tensor(target_id, dtype=torch.long),
                    'cluster_id': torch.tensor(cluster_id, dtype=torch.long)
                })
            
            if examples:
                return examples[np.random.randint(len(examples))]
        
        # Standard sequence-to-sequence training
        tokens = tokens[:self.max_length-2]  # Reserve space for BOS/EOS
        
        # Convert to IDs
        token_ids = [self.vocab.get('<s>', 3)]  # BOS
        token_ids.extend([self.vocab.get(token, self.vocab.get('<UNK>', 1)) for token in tokens])
        token_ids.append(self.vocab.get('</s>', 2))  # EOS
        
        # Create input/target sequences
        input_ids = token_ids[:-1]
        target_ids = token_ids[1:]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'cluster_id': torch.tensor(cluster_id, dtype=torch.long)
        }

def collate_batch(batch):
    """Collate function for DataLoader"""
    input_ids = [item['input_ids'] for item in batch]
    cluster_ids = torch.stack([item['cluster_id'] for item in batch])
    
    # Handle both single target and sequence targets
    if 'target_ids' in batch[0]:
        # Sequence-to-sequence training
        target_ids = [item['target_ids'] for item in batch]
        target_ids_padded = pad_sequence(target_ids, batch_first=True, padding_value=0)
    elif 'target_id' in batch[0]:
        # Single target (autocomplete)
        target_ids_padded = torch.stack([item['target_id'] for item in batch])
    else:
        # Fallback: create dummy targets from input
        target_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    
    # Pad input sequences
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    
    # Create category tensor
    batch_size, seq_len = input_ids_padded.shape
    category_ids = cluster_ids.unsqueeze(1).expand(batch_size, seq_len)
    
    return {
        'input_ids': input_ids_padded,
        'target_ids': target_ids_padded,
        'category_ids': category_ids
    }

class RealCADENCETrainer:
    """Trainer for real CADENCE models"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = torch.cuda.is_available()
        
        # Optimized settings for RTX 3050 4GB
        self.batch_size = 16 if torch.cuda.is_available() else 32
        self.gradient_accumulation_steps = 2
        self.max_grad_norm = 1.0
        
        logger.info(f"🔧 Trainer initialized:")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Mixed precision: {self.use_amp}")
        logger.info(f"   Batch size: {self.batch_size}")
        logger.info(f"   Gradient accumulation: {self.gradient_accumulation_steps}")
    
    def load_processed_data(self, data_dir: str = "processed_data"):
        """Load processed data"""
        data_path = Path(data_dir)
        
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
    
    def create_training_datasets(self, query_df: pd.DataFrame, product_df: pd.DataFrame, 
                               vocab: Dict[str, int], test_split: float = 0.2):
        """Create training and test datasets"""
        logger.info("📊 Creating training datasets...")
        
        # Prepare query data
        query_texts = query_df['processed_query'].tolist()
        query_clusters = query_df['cluster_id'].tolist()
        
        # Prepare product data
        product_texts = product_df['processed_title'].tolist()
        product_clusters = product_df['cluster_id'].tolist()
        
        # Split data
        query_split = int((1 - test_split) * len(query_texts))
        product_split = int((1 - test_split) * len(product_texts))
        
        # Query datasets
        query_train_texts = query_texts[:query_split]
        query_train_clusters = query_clusters[:query_split]
        query_test_texts = query_texts[query_split:]
        query_test_clusters = query_clusters[query_split:]
        
        # Product datasets  
        product_train_texts = product_texts[:product_split]
        product_train_clusters = product_clusters[:product_split]
        product_test_texts = product_texts[product_split:]
        product_test_clusters = product_clusters[product_split:]
        
        # Create datasets
        datasets = {
            'query_train': CADENCEDataset(query_train_texts, query_train_clusters, vocab, 
                                        max_length=64, is_autocomplete=True),
            'query_test': CADENCEDataset(query_test_texts, query_test_clusters, vocab, 
                                       max_length=64, is_autocomplete=True),
            'product_train': CADENCEDataset(product_train_texts, product_train_clusters, vocab, 
                                          max_length=128, is_autocomplete=False),
            'product_test': CADENCEDataset(product_test_texts, product_test_clusters, vocab, 
                                         max_length=128, is_autocomplete=False)
        }
        
        logger.info(f"✅ Created datasets:")
        for name, dataset in datasets.items():
            logger.info(f"   {name}: {len(dataset):,} samples")
        
        return datasets
    
    def train_model(self, model: RealCADENCEModel, train_dataset: Dataset, test_dataset: Dataset,
                   model_type: str, epochs: int = 5, save_path: str = None):
        """Train a CADENCE model"""
        logger.info(f"🚀 Training {model_type} model for {epochs} epochs...")
        
        # Create data loaders (no multiprocessing to avoid worker issues)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_batch,
            num_workers=0,  # Disable multiprocessing
            pin_memory=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_batch,
            num_workers=0,  # Disable multiprocessing
            pin_memory=False
        )
        
        # Setup optimizer and scaler
        optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scaler = GradScaler() if self.use_amp else None
        
        # Move model to device
        model.to(self.device)
        
        # Training loop
        best_loss = float('inf')
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            num_batches = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, batch in enumerate(pbar):
                # Move to device
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                target_ids = batch['target_ids'].to(self.device, non_blocking=True)
                category_ids = batch['category_ids'].to(self.device, non_blocking=True)
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        outputs = model(input_ids, category_ids, target_ids, model_type=model_type)
                        loss = outputs['loss'] / self.gradient_accumulation_steps
                else:
                    outputs = model(input_ids, category_ids, target_ids, model_type=model_type)
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
                
                total_loss += loss.item() * self.gradient_accumulation_steps
                num_batches += 1
                
                pbar.set_postfix({'loss': f"{loss.item() * self.gradient_accumulation_steps:.4f}"})
                
                # Memory cleanup
                if batch_idx % 50 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            avg_train_loss = total_loss / num_batches
            
            # Validation
            model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for batch in test_loader:
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
            
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
            
            logger.info(f"Epoch {epoch+1}/{epochs}:")
            logger.info(f"  Train Loss: {avg_train_loss:.4f}")
            logger.info(f"  Val Loss: {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < best_loss and save_path:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), save_path)
                logger.info(f"  ✅ Best model saved to {save_path}")
        
        return model
    
    def save_complete_model(self, model: RealCADENCEModel, vocab: Dict[str, int],
                          cluster_info: Dict[str, Any], model_name: str = "real_cadence"):
        """Save complete trained model with all metadata"""
        models_dir = Path("trained_models")
        models_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = models_dir / f"{model_name}.pt"
        torch.save(model.state_dict(), model_path)
        
        # Save vocabulary
        vocab_path = models_dir / f"{model_name}_vocab.pkl"
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f)
        
        # Save complete configuration
        config = {
            'model_config': model.config,
            'vocab_size': len(vocab),
            'num_categories': model.num_categories,
            'cluster_info': cluster_info,
            'training_date': datetime.now().isoformat(),
            'model_type': 'RealCADENCEModel',
            'total_parameters': model.count_parameters()
        }
        
        config_path = models_dir / f"{model_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"💾 Complete model saved:")
        logger.info(f"   Model: {model_path}")
        logger.info(f"   Vocabulary: {vocab_path}")
        logger.info(f"   Config: {config_path}")
        
        return model_path, vocab_path, config_path

def main():
    """Main training function"""
    logger.info("=" * 80)
    logger.info("🚀 REAL CADENCE MODEL TRAINING")
    logger.info("=" * 80)
    
    try:
        # Initialize trainer
        trainer = RealCADENCETrainer()
        
        # Load processed data
        logger.info("\nSTEP 1/5: Loading Processed Data")
        logger.info("-" * 50)
        query_df, product_df, vocab, cluster_info = trainer.load_processed_data()
        
        # Create datasets
        logger.info("\nSTEP 2/5: Creating Training Datasets")
        logger.info("-" * 50)
        datasets = trainer.create_training_datasets(query_df, product_df, vocab)
        
        # Calculate total categories
        total_categories = (cluster_info['total_query_clusters'] + 
                          cluster_info['total_product_clusters'])
        
        # Create model
        logger.info("\nSTEP 3/5: Creating CADENCE Model")
        logger.info("-" * 50)
        model_config = {
            'embedding_dim': 256,
            'hidden_dim': 512,
            'num_layers': 2,
            'dropout': 0.3,
            'attention_dim': 256
        }
        
        model = RealCADENCEModel(len(vocab), total_categories, model_config)
        
        # Train Query Language Model
        logger.info("\nSTEP 4/5: Training Query Language Model")
        logger.info("-" * 50)
        query_model_path = Path("trained_models") / "query_model_checkpoint.pt"
        Path("trained_models").mkdir(exist_ok=True)
        
        model = trainer.train_model(
            model, 
            datasets['query_train'],
            datasets['query_test'],
            model_type='query',
            epochs=3,
            save_path=str(query_model_path)
        )
        
        # Train Catalog Language Model
        logger.info("\nSTEP 4b/5: Training Catalog Language Model")
        logger.info("-" * 50)
        catalog_model_path = Path("trained_models") / "catalog_model_checkpoint.pt"
        
        model = trainer.train_model(
            model,
            datasets['product_train'], 
            datasets['product_test'],
            model_type='catalog',
            epochs=3,
            save_path=str(catalog_model_path)
        )
        
        # Save final model
        logger.info("\nSTEP 5/5: Saving Complete Model")
        logger.info("-" * 50)
        model_files = trainer.save_complete_model(model, vocab, cluster_info)
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info("📊 Training Results:")
        logger.info(f"   Model parameters: {model.count_parameters()/1e6:.1f}M")
        logger.info(f"   Vocabulary size: {len(vocab):,}")
        logger.info(f"   Total categories: {total_categories}")
        logger.info(f"   Query samples: {len(datasets['query_train']):,}")
        logger.info(f"   Product samples: {len(datasets['product_train']):,}")
        logger.info("")
        logger.info("🎯 Next Steps:")
        logger.info("   1. Run backend: python cadence_backend.py")
        logger.info("   2. Test APIs: http://localhost:8000/docs")
        logger.info("   3. Launch frontend: npm start (in frontend/)")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        logger.exception("Full error traceback:")
        raise

if __name__ == "__main__":
    main()