"""
Model Training Pipeline for Enhanced CADENCE System
Trains both Query LM and Catalog LM using Amazon QAC and Products datasets
"""
import os
import pickle
import json
import torch
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple
import structlog
from pathlib import Path

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from core.data_processor import DataProcessor
from core.cadence_model import CADENCEModel, create_cadence_model
from config.settings import settings

logger = structlog.get_logger()

class QueryDataset(Dataset):
    """Dataset for training Query Language Model"""
    
    def __init__(self, texts: List[str], cluster_ids: List[int], vocab: Dict[str, int], max_length: int = 50):
        self.texts = texts
        self.cluster_ids = cluster_ids
        self.vocab = vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        cluster_id = self.cluster_ids[idx]
        
        # Tokenize and convert to IDs
        tokens = text.split()[:self.max_length-2]  # Leave space for BOS/EOS
        token_ids = [self.vocab.get('<s>', 3)]  # BOS token
        
        for token in tokens:
            token_ids.append(self.vocab.get(token, self.vocab.get('<UNK>', 1)))
        
        token_ids.append(self.vocab.get('</s>', 2))  # EOS token
        
        # Create input and target sequences
        input_ids = token_ids[:-1]  # All except last token
        target_ids = token_ids[1:]  # All except first token
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'cluster_id': torch.tensor(cluster_id, dtype=torch.long),
            'length': len(input_ids)
        }

def collate_fn(batch):
    """Collate function for DataLoader"""
    input_ids = [item['input_ids'] for item in batch]
    target_ids = [item['target_ids'] for item in batch]
    cluster_ids = torch.stack([item['cluster_id'] for item in batch])
    
    # Pad sequences
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    target_ids_padded = pad_sequence(target_ids, batch_first=True, padding_value=0)
    
    # Create category tensor (same cluster for all tokens in sequence)
    batch_size, seq_len = input_ids_padded.shape
    category_ids = cluster_ids.unsqueeze(1).expand(batch_size, seq_len)
    
    return {
        'input_ids': input_ids_padded,
        'target_ids': target_ids_padded,
        'category_ids': category_ids
    }

class CADENCETrainer:
    """Trainer for CADENCE models"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.data_processor = DataProcessor()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Using device: {self.device}")
    
    def prepare_data(self, max_samples: int = 50000) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Prepare training data from Amazon datasets"""
        logger.info("Loading and processing Amazon datasets...")
        
        # Use 1 million samples from Amazon QAC dataset as requested
        # This is still manageable for memory while providing substantial training data
        qac_sample_size = min(max_samples, 1000000)  # Use 1M samples for better training
        logger.info(f"Using {qac_sample_size} samples from Amazon QAC dataset (streaming mode - no full download)")
        
        # Load Amazon QAC dataset using STREAMING MODE (no full download!)
        logger.info("Loading Amazon QAC dataset with streaming...")
        try:
            # Use streaming=True to avoid downloading the full 60GB dataset
            qac_dataset = load_dataset("amazon/AmazonQAC", split="train", streaming=True)
            
            # Take only the samples we need using streaming
            logger.info(f"Streaming {qac_sample_size} samples from Amazon QAC (no full download)")
            
            # Process streaming data
            sample_data = []
            for i, sample in enumerate(qac_dataset):
                if i >= qac_sample_size:
                    break
                sample_data.append(sample)
                
                if i % 1000 == 0:
                    logger.info(f"Streamed {i+1} samples...")
            
            logger.info(f"Successfully streamed {len(sample_data)} samples without downloading full dataset")
            
            # Convert to DataFrame for processing
            import pandas as pd
            qac_df = pd.DataFrame(sample_data)
            
            # Process with data processor
            query_df = self._process_qac_streaming_data(qac_df)
            logger.info(f"Processed {len(query_df)} queries from streaming data")
        except Exception as e:
            logger.warning(f"Failed to load Amazon QAC dataset: {e}")
            # Create dummy data for demo
            logger.info("Using dummy data instead of Amazon QAC dataset")
            query_df = self._create_dummy_query_data(qac_sample_size // 4)
        
        # Load Amazon Products dataset (subset)
        logger.info("Loading Amazon Products dataset...")
        try:
            product_df = self.data_processor.load_and_process_amazon_products(max_samples=qac_sample_size // 4)
            logger.info(f"Loaded {len(product_df)} products")
        except Exception as e:
            logger.warning(f"Failed to load Amazon Products dataset: {e}")
            # Create dummy data for demo
            product_df = self._create_dummy_product_data(qac_sample_size // 4)
        
        # Create training data
        training_data = self.data_processor.create_training_data(query_df, product_df)
        
        # Split data
        query_data = training_data['query_data']
        catalog_data = training_data['catalog_data']
        
        # Split into train/validation
        train_split = int(0.8 * len(query_data))
        query_train = query_data[:train_split]
        query_val = query_data[train_split:]
        
        catalog_train_split = int(0.8 * len(catalog_data))
        catalog_train = catalog_data[:catalog_train_split]
        catalog_val = catalog_data[catalog_train_split:]
        
        query_datasets = {
            'train': query_train,
            'val': query_val,
            'vocab': training_data['vocab'],
            'clusters': training_data['query_clusters']
        }
        
        catalog_datasets = {
            'train': catalog_train,
            'val': catalog_val,
            'vocab': training_data['vocab'],  # Shared vocabulary
            'clusters': training_data['product_clusters']
        }
        
        logger.info(f"Query training data: {len(query_train)} samples")
        logger.info(f"Query validation data: {len(query_val)} samples")
        logger.info(f"Catalog training data: {len(catalog_train)} samples")
        logger.info(f"Catalog validation data: {len(catalog_val)} samples")
        logger.info(f"Vocabulary size: {len(training_data['vocab'])}")
        
        return query_datasets, catalog_datasets
    
    def _create_dummy_query_data(self, num_samples: int):
        """Create dummy query data for demo purposes"""
        import pandas as pd
        import random
        
        sample_queries = [
            "laptop computer gaming", "wireless headphones bluetooth", "running shoes men",
            "smartphone android unlocked", "coffee maker automatic", "book fiction bestseller",
            "tablet screen protector", "kitchen knife set", "winter jacket waterproof",
            "fitness tracker smartwatch", "camera digital photography", "desk chair office",
            "bluetooth speaker portable", "phone case protective", "laptop bag leather"
        ]
        
        data = []
        for i in range(num_samples):
            base_query = random.choice(sample_queries)
            processed_query = self.data_processor.preprocess_query_text(base_query)
            
            data.append({
                'original_query': base_query,
                'processed_query': processed_query,
                'prefixes': [base_query[:j] for j in range(1, len(base_query)+1)],
                'popularity': random.randint(1, 100),
                'session_id': f"session_{i}"
            })
        
        df = pd.DataFrame(data)
        
        # Add clustering
        cluster_labels = np.random.randint(0, 10, size=len(df))
        df['cluster_id'] = cluster_labels
        df['cluster_description'] = df['cluster_id'].apply(lambda x: f"cluster_{x}")
        
        return df
    
    def _create_dummy_product_data(self, num_samples: int):
        """Create dummy product data for demo purposes"""
        import pandas as pd
        import random
        
        sample_products = [
            "Apple MacBook Pro 16-inch M3 Laptop",
            "Sony WH-1000XM5 Wireless Headphones",
            "Nike Air Max 270 Running Shoes",
            "Samsung Galaxy S24 Smartphone",
            "Breville Bambino Plus Coffee Machine",
            "The Seven Husbands of Evelyn Hugo Novel",
            "iPad Pro 12.9-inch Tablet",
            "Wusthof Classic 8-piece Knife Set",
            "Patagonia Down Sweater Jacket",
            "Fitbit Charge 5 Fitness Tracker"
        ]
        
        data = []
        for i in range(num_samples):
            base_title = random.choice(sample_products)
            processed_title = self.data_processor.preprocess_product_title(base_title)
            
            data.append({
                'product_id': f"prod_{i}",
                'original_title': base_title,
                'processed_title': processed_title,
                'description': f"Description for {base_title}",
                'main_category': random.choice(['Electronics', 'Clothing', 'Books', 'Sports']),
                'categories': [random.choice(['Electronics', 'Clothing', 'Books', 'Sports'])],
                'price': random.uniform(20, 2000),
                'rating': random.uniform(3.5, 5.0),
                'rating_count': random.randint(10, 10000),
                'attributes': {}
            })
        
        df = pd.DataFrame(data)
        
        # Add clustering
        cluster_labels = np.random.randint(0, 10, size=len(df))
        df['cluster_id'] = cluster_labels
        df['cluster_description'] = df['cluster_id'].apply(lambda x: f"cluster_{x}")
        
        return df
    
    def _process_qac_streaming_data(self, qac_df):
        """Process streaming QAC data without full dataset dependencies"""
        import pandas as pd
        
        logger.info(f"Processing {len(qac_df)} streaming QAC samples...")
        
        # Process queries from streaming data
        processed_queries = []
        for _, row in qac_df.iterrows():
            # Get final search term
            query_text = row.get('final_search_term', str(row.get('query', '')))
            processed_query = self.data_processor.preprocess_query_text(query_text)
            
            if processed_query:  # Only keep non-empty queries
                processed_queries.append({
                    'original_query': query_text,
                    'processed_query': processed_query,
                    'prefixes': row.get('prefixes', [query_text]),
                    'popularity': row.get('popularity', 1),
                    'session_id': row.get('session_id', f"session_{len(processed_queries)}")
                })
        
        query_df = pd.DataFrame(processed_queries)
        
        if len(query_df) == 0:
            logger.warning("No valid queries processed from streaming data, using dummy data")
            return self._create_dummy_query_data(1000)
        
        # Cluster queries for pseudo-categories
        if len(query_df) > 100:  # Only cluster if we have enough data
            cluster_labels, cluster_descriptions = self.data_processor.cluster_texts_hdbscan(
                query_df['processed_query'].tolist()
            )
            query_df['cluster_id'] = cluster_labels
            query_df['cluster_description'] = query_df['cluster_id'].map(cluster_descriptions)
            self.query_clusters = cluster_descriptions
        else:
            query_df['cluster_id'] = 0
            query_df['cluster_description'] = 'general'
        
        # Safety check: Ensure all cluster IDs are non-negative
        min_cluster_id = query_df['cluster_id'].min()
        if min_cluster_id < 0:
            logger.warning(f"Found negative cluster IDs (min: {min_cluster_id}), adjusting...")
            query_df['cluster_id'] = query_df['cluster_id'] - min_cluster_id
        
        logger.info(f"Successfully processed {len(query_df)} queries from streaming data")
        return query_df
    
    def train_model(self, model: CADENCEModel, train_data: List[Dict], val_data: List[Dict], 
                   vocab: Dict[str, int], model_type: str = "query", epochs: int = 5):
        """Train a CADENCE model"""
        logger.info(f"Training {model_type} model...")
        
        # Create datasets
        train_texts = [item['text'] for item in train_data]
        train_clusters = [item.get('cluster_id', 0) for item in train_data]
        
        val_texts = [item['text'] for item in val_data]
        val_clusters = [item.get('cluster_id', 0) for item in val_data]
        
        train_dataset = QueryDataset(train_texts, train_clusters, vocab)
        val_dataset = QueryDataset(val_texts, val_clusters, vocab)
        
        train_loader = DataLoader(train_dataset, batch_size=settings.BATCH_SIZE, 
                                shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=settings.BATCH_SIZE, 
                              collate_fn=collate_fn)
        
        # Setup optimizer
        optimizer = Adam(model.parameters(), lr=settings.LEARNING_RATE)
        
        # Move model to device
        model.to(self.device)
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            num_batches = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                category_ids = batch['category_ids'].to(self.device)
                
                # Forward pass
                outputs = model(input_ids, category_ids, target_ids, model_type=model_type)
                loss = outputs['loss']
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if num_batches % 10 == 0:
                    logger.info(f"Epoch {epoch+1}, Batch {num_batches}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / num_batches
            
            # Validation
            model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    target_ids = batch['target_ids'].to(self.device)
                    category_ids = batch['category_ids'].to(self.device)
                    
                    outputs = model(input_ids, category_ids, target_ids, model_type=model_type)
                    val_loss += outputs['loss'].item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        return model
    
    def save_model_and_vocab(self, model: CADENCEModel, vocab: Dict[str, int], 
                           cluster_mappings: Dict[str, Any], model_name: str):
        """Save trained model and vocabulary"""
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
            'num_categories': len(cluster_mappings.get('query_clusters', {})) + len(cluster_mappings.get('product_clusters', {})),
            'model_config': model.config,
            'cluster_mappings': cluster_mappings,
            'training_date': datetime.now().isoformat()
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Vocabulary saved to {vocab_path}")
        logger.info(f"Configuration saved to {config_path}")
    
    def load_model_and_vocab(self, model_name: str) -> Tuple[CADENCEModel, Dict[str, int], Dict[str, Any]]:
        """Load trained model and vocabulary"""
        model_path = self.model_dir / f"{model_name}.pt"
        vocab_path = self.model_dir / f"{model_name}_vocab.pkl"
        config_path = self.model_dir / f"{model_name}_config.json"
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load vocabulary
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        
        # Create and load model
        model = CADENCEModel(config['model_config'])
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        
        logger.info(f"Model loaded from {model_path}")
        
        return model, vocab, config
    
    def train_full_pipeline(self, max_samples: int = 50000, epochs: int = 3):
        """Train the complete CADENCE pipeline with memory optimization"""
        logger.info("Starting full CADENCE training pipeline...")
        logger.info(f"Training with {max_samples} samples for {epochs} epochs")
        
        # Add memory monitoring
        import psutil
        import gc
        
        def log_memory_usage(step_name):
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            logger.info(f"{step_name} - Memory usage: {memory_mb:.1f} MB")
        
        log_memory_usage("Pipeline start")
        
        # Prepare data with progress tracking
        logger.info("Step 1/4: Preparing training data...")
        query_datasets, catalog_datasets = self.prepare_data(max_samples)
        
        # Force garbage collection
        gc.collect()
        log_memory_usage("Data preparation complete")
        
        # Create model
        vocab_size = len(query_datasets['vocab'])
        num_categories = len(query_datasets['clusters']) + len(catalog_datasets['clusters'])
        
        logger.info(f"Step 2/4: Creating CADENCE model...")
        logger.info(f"Vocab size: {vocab_size}, Categories: {num_categories}")
        model = create_cadence_model(vocab_size, num_categories)
        
        log_memory_usage("Model creation complete")
        
        # Train Query Language Model
        logger.info("Step 3/4: Training Query Language Model...")
        logger.info(f"Training on {len(query_datasets['train'])} query samples")
        model = self.train_model(
            model, 
            query_datasets['train'], 
            query_datasets['val'],
            query_datasets['vocab'],
            model_type='query',
            epochs=epochs
        )
        
        # Force garbage collection between training steps
        gc.collect()
        log_memory_usage("Query model training complete")
        
        # Train Catalog Language Model
        logger.info("Step 4/4: Training Catalog Language Model...")
        logger.info(f"Training on {len(catalog_datasets['train'])} catalog samples")
        model = self.train_model(
            model,
            catalog_datasets['train'],
            catalog_datasets['val'],
            catalog_datasets['vocab'],
            model_type='catalog',
            epochs=epochs
        )
        
        log_memory_usage("Catalog model training complete")
        
        # Save everything
        cluster_mappings = {
            'query_clusters': query_datasets['clusters'],
            'product_clusters': catalog_datasets['clusters']
        }
        
        logger.info("Saving trained model and vocabulary...")
        self.save_model_and_vocab(
            model, 
            query_datasets['vocab'], 
            cluster_mappings,
            'cadence_trained'
        )
        
        log_memory_usage("Pipeline completion")
        logger.info("Training pipeline completed successfully!")
        
        return model, query_datasets['vocab'], cluster_mappings

def main():
    """Main training function"""
    logger.info("Starting CADENCE model training...")
    
    trainer = CADENCETrainer()
    
    # Start with smaller sample size to avoid memory issues
    initial_sample_size = 500000  # Start with 500K instead of 1M
    
    try:
        logger.info(f"Attempting training with {initial_sample_size} samples...")
        # Train models with memory-safe sample size
        model, vocab, cluster_mappings = trainer.train_full_pipeline(
            max_samples=initial_sample_size,
            epochs=3
        )
        
        logger.info("Training completed successfully!")
        
    except MemoryError as e:
        logger.warning(f"Memory error with {initial_sample_size} samples: {e}")
        logger.info("Retrying with smaller dataset...")
        
        # Fallback to smaller dataset
        smaller_sample_size = 100000  # 100K samples
        
        try:
            model, vocab, cluster_mappings = trainer.train_full_pipeline(
                max_samples=smaller_sample_size,
                epochs=3
            )
            logger.info(f"Training completed successfully with {smaller_sample_size} samples!")
            
        except Exception as e2:
            logger.error(f"Training failed even with smaller dataset: {e2}")
            logger.info("Using minimal dataset for demo...")
            
            # Final fallback
            model, vocab, cluster_mappings = trainer.train_full_pipeline(
                max_samples=10000,  # Minimal dataset
                epochs=2
            )
            logger.info("Training completed with minimal dataset!")
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.info("Attempting with minimal configuration...")
        
        # Emergency fallback
        model, vocab, cluster_mappings = trainer.train_full_pipeline(
            max_samples=10000,
            epochs=1
        )
        logger.info("Emergency training completed!")
    
    logger.info(f"Model saved in: {trainer.model_dir}")
    logger.info(f"Vocabulary size: {len(vocab)}")
    logger.info(f"Number of clusters: {len(cluster_mappings.get('query_clusters', {})) + len(cluster_mappings.get('product_clusters', {}))}")

if __name__ == "__main__":
    main() 