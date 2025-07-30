"""
Enhanced CADENCE Model Implementation
GRU-MN with Category Constraints and Self-Attention for Query Generation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import math
import structlog

logger = structlog.get_logger()

class GRUMemoryNetworkCell(nn.Module):
    """
    GRU Memory Network Cell with Self-Attention
    Based on CADENCE paper implementation
    """
    
    def __init__(self, input_size: int, hidden_size: int, attention_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        
        # Standard GRU gates
        self.weight_ih = nn.Linear(input_size, 3 * hidden_size)
        self.weight_hh = nn.Linear(hidden_size, 3 * hidden_size)
        
        # Attention mechanism weights
        self.W_h = nn.Linear(hidden_size, attention_size)
        self.W_x = nn.Linear(input_size, attention_size)
        self.W_tilde_h = nn.Linear(hidden_size, attention_size)
        self.v = nn.Linear(attention_size, 1)
        
    def forward(self, input_tensor: torch.Tensor, hidden: torch.Tensor, 
                memory_tape: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through GRU-MN cell
        """
        batch_size = input_tensor.size(0)
        
        # Compute attention if memory tape exists
        if memory_tape is not None and memory_tape.size(1) > 0:
            # Compute attention scores
            seq_len = memory_tape.size(1)
            
            # Expand inputs for attention computation
            h_expanded = self.W_h(memory_tape)  # [batch, seq_len, attention_size]
            x_expanded = self.W_x(input_tensor).unsqueeze(1).expand(-1, seq_len, -1)
            tilde_h_expanded = self.W_tilde_h(hidden).unsqueeze(1).expand(-1, seq_len, -1)
            
            # Compute attention scores
            attention_input = torch.tanh(h_expanded + x_expanded + tilde_h_expanded)
            attention_scores = self.v(attention_input).squeeze(-1)  # [batch, seq_len]
            attention_weights = F.softmax(attention_scores, dim=1)
            
            # Compute aggregated hidden state
            aggregated_hidden = torch.sum(attention_weights.unsqueeze(-1) * memory_tape, dim=1)
        else:
            aggregated_hidden = hidden
        
        # Standard GRU computation with aggregated hidden state
        gi = self.weight_ih(input_tensor)
        gh = self.weight_hh(aggregated_hidden)
        
        i_reset, i_update, i_new = gi.chunk(3, 1)
        h_reset, h_update, h_new = gh.chunk(3, 1)
        
        reset_gate = torch.sigmoid(i_reset + h_reset)
        update_gate = torch.sigmoid(i_update + h_update)
        new_gate = torch.tanh(i_new + reset_gate * h_new)
        
        new_hidden = (1 - update_gate) * new_gate + update_gate * aggregated_hidden
        
        return new_hidden, aggregated_hidden

class CategoryConstrainedGRUMN(nn.Module):
    """
    Multi-layer GRU-MN with Category Constraints
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dims: List[int], 
                 attention_dims: List[int], num_categories: int, dropout: float = 0.5):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.attention_dims = attention_dims
        self.num_categories = num_categories
        self.num_layers = len(hidden_dims)
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Category embedding
        self.category_embedding = nn.Embedding(num_categories, embedding_dim)
        
        # GRU-MN layers
        self.gru_cells = nn.ModuleList()
        input_size = embedding_dim * 2  # word + category
        
        for i, (hidden_dim, attention_dim) in enumerate(zip(hidden_dims, attention_dims)):
            self.gru_cells.append(GRUMemoryNetworkCell(input_size, hidden_dim, attention_dim))
            input_size = hidden_dim
        
        # Intermediate context layer
        self.intermediate_context = nn.Linear(hidden_dims[-1] + embedding_dim, 500)
        
        # Output layer
        self.output_projection = nn.Linear(500, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Memory tapes for each layer
        self.memory_tapes = [None] * self.num_layers
        
    def forward(self, input_ids: torch.Tensor, category_ids: torch.Tensor, 
                hidden_states: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the model
        """
        batch_size, seq_len = input_ids.size()
        
        # Initialize hidden states if not provided
        if hidden_states is None:
            hidden_states = [torch.zeros(batch_size, dim, device=input_ids.device) 
                           for dim in self.hidden_dims]
        
        # Embed inputs
        word_embeds = self.embedding(input_ids)  # [batch, seq_len, embed_dim]
        category_embeds = self.category_embedding(category_ids)  # [batch, seq_len, embed_dim]
        
        # Concatenate word and category embeddings
        input_embeds = torch.cat([word_embeds, category_embeds], dim=-1)
        
        outputs = []
        
        # Process each time step
        for t in range(seq_len):
            x_t = input_embeds[:, t, :]  # [batch, embed_dim * 2]
            
            # Pass through each layer
            layer_input = x_t
            new_hidden_states = []
            
            for layer_idx, gru_cell in enumerate(self.gru_cells):
                # Add category constraint to input
                category_t = category_embeds[:, t, :]
                layer_input_with_category = F.relu(layer_input + category_t)
                
                # Forward through GRU-MN cell
                new_hidden, _ = gru_cell(
                    layer_input_with_category, 
                    hidden_states[layer_idx],
                    self.memory_tapes[layer_idx]
                )
                
                # Update memory tape
                if self.memory_tapes[layer_idx] is None:
                    self.memory_tapes[layer_idx] = new_hidden.unsqueeze(1)
                else:
                    self.memory_tapes[layer_idx] = torch.cat([
                        self.memory_tapes[layer_idx], 
                        new_hidden.unsqueeze(1)
                    ], dim=1)
                
                new_hidden_states.append(new_hidden)
                layer_input = self.dropout(new_hidden)
            
            hidden_states = new_hidden_states
            
            # Intermediate context layer with category information
            context_input = torch.cat([hidden_states[-1], category_embeds[:, t, :]], dim=-1)
            context_output = torch.tanh(self.intermediate_context(context_input))
            
            # Output projection
            output = self.output_projection(context_output)
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=1)  # [batch, seq_len, vocab_size]
        
        return outputs, hidden_states
    
    def reset_memory(self):
        """Reset memory tapes for new sequence"""
        self.memory_tapes = [None] * self.num_layers

class CADENCEModel(nn.Module):
    """
    Complete CADENCE Model for Query Generation
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.vocab_size = config['vocab_size']
        self.num_categories = config['num_categories']
        
        # Query Language Model
        self.query_lm = CategoryConstrainedGRUMN(
            vocab_size=self.vocab_size,
            embedding_dim=config['embedding_dim'],
            hidden_dims=config['hidden_dims'],
            attention_dims=config['attention_dims'],
            num_categories=self.num_categories,
            dropout=config['dropout']
        )
        
        # Catalog Language Model (same architecture)
        self.catalog_lm = CategoryConstrainedGRUMN(
            vocab_size=self.vocab_size,
            embedding_dim=config['embedding_dim'],
            hidden_dims=config['hidden_dims'],
            attention_dims=config['attention_dims'],
            num_categories=self.num_categories,
            dropout=config['dropout']
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        
    def forward(self, input_ids: torch.Tensor, category_ids: torch.Tensor, 
                target_ids: torch.Tensor, model_type: str = 'query') -> Dict[str, torch.Tensor]:
        """
        Forward pass for training
        """
        if model_type == 'query':
            outputs, hidden_states = self.query_lm(input_ids, category_ids)
        else:
            outputs, hidden_states = self.catalog_lm(input_ids, category_ids)
        
        # Calculate loss
        loss = self.criterion(outputs.view(-1, self.vocab_size), target_ids.view(-1))
        
        return {
            'logits': outputs,
            'loss': loss,
            'hidden_states': hidden_states
        }
    
    def generate(self, start_tokens: List[int], category_id: int, max_length: int = 50, 
                 model_type: str = 'query', temperature: float = 1.0) -> List[int]:
        """
        Generate query using the model
        """
        self.eval()
        
        if model_type == 'query':
            model = self.query_lm
        else:
            model = self.catalog_lm
        
        model.reset_memory()
        
        with torch.no_grad():
            device = next(self.parameters()).device
            
            # Initialize
            current_tokens = start_tokens.copy()
            hidden_states = None
            
            for _ in range(max_length):
                # Prepare input
                input_tensor = torch.tensor([current_tokens], device=device)
                category_tensor = torch.full_like(input_tensor, category_id, device=device)
                
                # Forward pass
                outputs, hidden_states = model(input_tensor, category_tensor, hidden_states)
                
                # Get next token probabilities
                next_token_logits = outputs[0, -1, :] / temperature
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(next_token_probs, 1).item()
                
                # Check for end token
                if next_token == 2:  # </s> token
                    break
                
                current_tokens.append(next_token)
            
            return current_tokens

class DynamicBeamSearch:
    """
    Dynamic Beam Search for Diverse Query Generation
    """
    
    def __init__(self, model: CADENCEModel, vocab: Dict[str, int], 
                 beam_width: int = 5, max_length: int = 50):
        self.model = model
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.beam_width = beam_width
        self.max_length = max_length
    
    def search(self, start_tokens: List[str], category_id: int, model_type: str = 'query',
               p1_threshold: float = 0.3, p2_threshold: float = 0.4) -> List[str]:
        """
        Perform dynamic beam search with diversity
        """
        self.model.eval()
        
        # Convert start tokens to IDs
        start_token_ids = [self.vocab.get(token, self.vocab['<UNK>']) for token in start_tokens]
        
        # First level: high diversity
        first_level_candidates = self._get_diverse_next_tokens(
            start_token_ids, category_id, model_type, p1_threshold
        )
        
        # Second level: moderate diversity
        second_level_candidates = []
        for candidate in first_level_candidates:
            next_candidates = self._get_diverse_next_tokens(
                candidate, category_id, model_type, p2_threshold
            )
            second_level_candidates.extend(next_candidates)
        
        # Final level: beam search
        final_candidates = []
        for candidate in second_level_candidates:
            beam_results = self._beam_search_from_prefix(
                candidate, category_id, model_type
            )
            final_candidates.extend(beam_results)
        
        # Convert back to strings and remove duplicates
        result_strings = []
        for candidate in final_candidates:
            tokens = [self.inv_vocab.get(token_id, '<UNK>') for token_id in candidate]
            query_string = ' '.join(tokens).replace('<s>', '').replace('</s>', '').strip()
            if query_string and query_string not in result_strings:
                result_strings.append(query_string)
        
        return result_strings[:10]  # Return top 10
    
    def _get_diverse_next_tokens(self, prefix: List[int], category_id: int, 
                                model_type: str, threshold: float) -> List[List[int]]:
        """Get diverse next tokens above threshold"""
        with torch.no_grad():
            device = next(self.model.parameters()).device
            
            # Prepare input
            input_tensor = torch.tensor([prefix], device=device)
            category_tensor = torch.full_like(input_tensor, category_id, device=device)
            
            # Get model
            if model_type == 'query':
                model = self.model.query_lm
            else:
                model = self.model.catalog_lm
            
            model.reset_memory()
            outputs, _ = model(input_tensor, category_tensor)
            
            # Get probabilities for next token
            next_token_probs = F.softmax(outputs[0, -1, :], dim=-1)
            
            # Get tokens above threshold
            candidates = []
            for token_id, prob in enumerate(next_token_probs):
                if prob.item() > threshold:
                    candidates.append(prefix + [token_id])
            
            return candidates[:20]  # Limit for efficiency
    
    def _beam_search_from_prefix(self, prefix: List[int], category_id: int, 
                                model_type: str) -> List[List[int]]:
        """Standard beam search from given prefix"""
        with torch.no_grad():
            device = next(self.model.parameters()).device
            
            # Initialize beam
            beam = [(prefix, 0.0)]  # (sequence, score)
            
            for _ in range(self.max_length - len(prefix)):
                candidates = []
                
                for sequence, score in beam:
                    if sequence[-1] == 2:  # End token
                        candidates.append((sequence, score))
                        continue
                    
                    # Get next token probabilities
                    input_tensor = torch.tensor([sequence], device=device)
                    category_tensor = torch.full_like(input_tensor, category_id, device=device)
                    
                    if model_type == 'query':
                        model = self.model.query_lm
                    else:
                        model = self.model.catalog_lm
                    
                    model.reset_memory()
                    outputs, _ = model(input_tensor, category_tensor)
                    
                    next_token_probs = F.softmax(outputs[0, -1, :], dim=-1)
                    
                    # Add top tokens to candidates
                    top_tokens = torch.topk(next_token_probs, self.beam_width)
                    
                    for token_id, prob in zip(top_tokens.indices, top_tokens.values):
                        new_sequence = sequence + [token_id.item()]
                        new_score = score + torch.log(prob).item()
                        candidates.append((new_sequence, new_score))
                
                # Keep top candidates
                candidates.sort(key=lambda x: x[1], reverse=True)
                beam = candidates[:self.beam_width]
                
                # Check if all sequences ended
                if all(seq[-1] == 2 for seq, _ in beam):
                    break
            
            return [seq for seq, _ in beam]

def create_cadence_model(vocab_size: int, num_categories: int) -> CADENCEModel:
    """
    Factory function to create CADENCE model with default config
    """
    config = {
        'vocab_size': vocab_size,
        'num_categories': num_categories,
        'embedding_dim': 256,
        'hidden_dims': [2000, 1500, 1000],
        'attention_dims': [1000, 750, 500],
        'dropout': 0.8
    }
    
    return CADENCEModel(config) 