import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast
import json
import os
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 30
    block_size: int = 1024
    embed_dim: int = 768
    n_layer: int = 12
    n_head: int = 12
    ff_dim: int = 3072
    dropout: float = 0.1

class ProteinTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.transformer = nn.Transformer(
            d_model=config.embed_dim,
            nhead=config.n_head,
            num_encoder_layers=config.n_layer,
            num_decoder_layers=config.n_layer,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout
        )
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.fc_out = nn.Linear(config.embed_dim, config.vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x, x)
        return self.fc_out(x)

class ProteinTokenizer:
    def __init__(self):
        # Standard amino acid vocabulary
        self.amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                           'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                           'X', 'B', 'J', 'Z', 'U', 'O', '*', '-']
        
        # Special tokens
        self.pad_token = '[PAD]'
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
        self.mask_token = '[MASK]'
        self.unk_token = '[UNK]'
        
        # Create vocabulary
        self.vocab = {token: i for i, token in enumerate(
            [self.pad_token, self.cls_token, self.sep_token, self.mask_token, self.unk_token] +
            self.amino_acids
        )}
        self.inv_vocab = {i: token for token, i in self.vocab.items()}
        
        # Set token IDs
        self.pad_token_id = self.vocab[self.pad_token]
        self.cls_token_id = self.vocab[self.cls_token]
        self.sep_token_id = self.vocab[self.sep_token]
        self.mask_token_id = self.vocab[self.mask_token]
        self.unk_token_id = self.vocab[self.unk_token]
        
        self.vocab_size = len(self.vocab)

    def encode_plus(self, 
                   sequences: list, 
                   add_special_tokens: bool = True,
                   padding: bool = True,
                   return_tensors: str = None,
                   is_split_into_words: bool = False) -> Dict[str, torch.Tensor]:
        """
        Encode a list of sequences into token IDs with optional special tokens and padding.
        """
        if not is_split_into_words:
            sequences = [list(seq) for seq in sequences]
        
        # Add special tokens if requested
        if add_special_tokens:
            encoded = [[self.cls_token_id] + 
                      [self.vocab.get(token, self.unk_token_id) for token in seq] +
                      [self.sep_token_id] for seq in sequences]
        else:
            encoded = [[self.vocab.get(token, self.unk_token_id) for token in seq]
                      for seq in sequences]
        
        # Get maximum length for padding
        if padding:
            max_len = max(len(seq) for seq in encoded)
            attention_mask = [[1] * len(seq) + [0] * (max_len - len(seq)) for seq in encoded]
            encoded = [seq + [self.pad_token_id] * (max_len - len(seq)) for seq in encoded]
        else:
            attention_mask = [[1] * len(seq) for seq in encoded]
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            encoded = torch.tensor(encoded)
            attention_mask = torch.tensor(attention_mask)
        
        return {
            "input_ids": encoded,
            "attention_mask": attention_mask
        }

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """
        Decode a sequence of token IDs back into amino acid sequence.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        tokens = [self.inv_vocab[id] for id in token_ids]
        
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in 
                     {self.pad_token, self.cls_token, self.sep_token, self.mask_token}]
        
        return ''.join(tokens)

class AnkhLoader:
    """Helper class for loading and managing protein language models"""
    
    @staticmethod
    def load_base_model() -> Tuple[Optional[nn.Module], ProteinTokenizer]:
        """
        Load the base protein language model and tokenizer
        Returns:
            Tuple of (model, tokenizer)
        """
        tokenizer = ProteinTokenizer()
        return None, tokenizer
    
    @staticmethod
    def save_checkpoint(model: nn.Module, 
                       optimizer: torch.optim.Optimizer,
                       config: ModelConfig,
                       epoch: int,
                       loss: float,
                       path: str):
        """
        Save a model checkpoint including state dict, optimizer state, and metadata
        """
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config.__dict__,
            'epoch': epoch,
            'loss': loss
        }
        torch.save(checkpoint, path)
    
    @staticmethod    
    def load_checkpoint(path: str, 
                       model: Optional[nn.Module] = None,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       device: str = 'cpu') -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Load a model checkpoint and return the model and metadata
        Handles different checkpoint formats and provides detailed error messages
        """
        try:
            checkpoint = torch.load(path, map_location=device)
            
            # Determine checkpoint format
            if isinstance(checkpoint, dict):
                # If checkpoint is already a state dict (direct model weights)
                if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                    state_dict = checkpoint
                    metadata = {'config': None, 'epoch': None, 'loss': None}
                # If checkpoint has our expected structure
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    metadata = {
                        'config': ModelConfig(**checkpoint['config']) if 'config' in checkpoint else None,
                        'epoch': checkpoint.get('epoch'),
                        'loss': checkpoint.get('loss')
                    }
                # If checkpoint has a different structure but contains state dict
                else:
                    # Try to find the state dict in the checkpoint
                    state_dict_keys = [k for k in checkpoint.keys() if 'state_dict' in k.lower()]
                    if state_dict_keys:
                        state_dict = checkpoint[state_dict_keys[0]]
                    else:
                        # Assume the checkpoint itself is the state dict
                        state_dict = checkpoint
                    metadata = {'config': None, 'epoch': None, 'loss': None}
            else:
                raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}")

            # Load state dict into model
            if model is not None:
                # Try to load state dict
                try:
                    model.load_state_dict(state_dict)
                except RuntimeError as e:
                    print(f"Warning: Failed to load state dict exactly. Attempting flexible loading...")
                    # Try flexible loading
                    model_dict = model.state_dict()
                    # Filter out size mismatch and unnecessary keys
                    filtered_state_dict = {k: v for k, v in state_dict.items() 
                                         if k in model_dict and v.size() == model_dict[k].size()}
                    # Update model dict and load
                    model_dict.update(filtered_state_dict)
                    model.load_state_dict(model_dict)
                    print(f"Loaded {len(filtered_state_dict)}/{len(state_dict)} layers")

            # Load optimizer state if provided
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except:
                    print("Warning: Failed to load optimizer state")

            return model, metadata

        except Exception as e:
            print(f"Error loading checkpoint from {path}")
            print(f"Error details: {str(e)}")
            print("\nCheckpoint contents:")
            try:
                checkpoint = torch.load(path, map_location=device)
                if isinstance(checkpoint, dict):
                    print("Keys in checkpoint:", checkpoint.keys())
                    for k, v in checkpoint.items():
                        print(f"{k}: {type(v)}")
                else:
                    print(f"Checkpoint type: {type(checkpoint)}")
            except:
                print("Could not inspect checkpoint contents")
            raise

def get_model_config(model_path: str) -> ModelConfig:
    """
    Extract model configuration from a saved checkpoint
    """
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # If checkpoint contains our config format
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            return ModelConfig(**checkpoint['config'])
            
        # Try to infer config from model architecture
        config_dict = {
            'block_size': None,
            'n_layer': None,
            'embed_dim': None,
            'n_head': None,
            'vocab_size': 30  # Default for protein sequences
        }
        
        # Try to extract dimensions from state dict
        if isinstance(checkpoint, dict):
            for key, value in checkpoint.items():
                if isinstance(value, torch.Tensor):
                    if 'embeddings' in key.lower() and len(value.shape) == 2:
                        config_dict['embed_dim'] = value.shape[1]
                    elif 'attention' in key.lower() and len(value.shape) == 4:
                        config_dict['n_head'] = value.shape[1]
                    elif 'position' in key.lower() and len(value.shape) == 3:
                        config_dict['block_size'] = value.shape[1]
        
        # Count number of layers
        if isinstance(checkpoint, dict):
            layer_count = sum(1 for k in checkpoint.keys() if 'layer.' in k or 'block.' in k)
            config_dict['n_layer'] = layer_count // 4  # Approximate number of transformer blocks
        
        # Fill in missing values with defaults
        config_dict = {k: v if v is not None else 512 for k, v in config_dict.items()}
        
        return ModelConfig(**config_dict)
        
    except Exception as e:
        print(f"Error loading config from {model_path}: {str(e)}")
        # Return default config
        return ModelConfig()

# Create singleton instance
ankh = AnkhLoader()

import torch
from index import ankh, ModelConfig

# Create model first
config = ModelConfig()  # This will use default values
model = ProteinTransformer(config)

# Load checkpoint with detailed error handling
try:
    model, metadata = ankh.load_checkpoint(
        "out/model_final.pt",
        model=model,
        device='cpu'  # or 'cuda' if using GPU
    )
    print("Successfully loaded checkpoint")
    if metadata['config']:
        print("Config:", metadata['config'].__dict__)
except Exception as e:
    print(f"Failed to load checkpoint: {str(e)}")