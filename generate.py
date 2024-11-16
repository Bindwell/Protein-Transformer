import math
import torch
import torch.nn.functional as F
import argparse
import os
from index import (
    ProteinTransformer, 
    ModelConfig,
    ProteinTokenizer
)

def get_model_config(model_path):
    """Extract configuration from saved model"""
    state_dict = torch.load(model_path, map_location='cpu')
    # Get block size from the attention bias matrix shape
    block_size = state_dict['transformer.h.0.attn.bias'].shape[-1]
    # Count number of transformer layers
    layer_count = sum(1 for k in state_dict if 'transformer.h' in k and 'ln_1.weight' in k)
    # Get embedding dimension from layer norm
    embed_dim = state_dict['transformer.ln_f.weight'].shape[0]
    config = ModelConfig(
        block_size=block_size,
        n_layer=layer_count,
        embed_dim=embed_dim,
        n_head=4,
        vocab_size=22  # Added vocab_size for protein tokens
    )
    print(f"Extracted config from model:")
    print(f"  block_size: {config.block_size}")
    print(f"  n_layer: {config.n_layer}")
    print(f"  embed_dim: {config.embed_dim}")
    print(f"  n_head: {config.n_head}")
    return config
@torch.no_grad()
def generate_protein(model, tokenizer, max_length=200, temperature=0.8, top_k=None, device='cpu'):
    model.eval()
    
    # Start with just a batch dimension
    x = torch.zeros((1, 1), dtype=torch.long, device=device)
    attention_mask = torch.ones((1, 1), device=device)
    
    generated_sequence = []
    
    for _ in range(max_length):
        # Get predictions
        logits, _ = model(x, attention_mask)
        next_token_logits = logits[:, -1, :] / temperature
        
        # Apply top-k sampling
        if top_k is not None:
            v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
            indices_to_remove = next_token_logits < v[:, [-1]]
            next_token_logits[indices_to_remove] = float('-inf')
        
        # Sample next token
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Stop if we sample padding token
        if next_token.item() == tokenizer.pad_token_id:
            break
            
        # Append token
        x = torch.cat([x, next_token], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones(1, 1, device=device)], dim=1)
        
        # For the first position, force it to be Methionine (token_id = 11)
        if len(generated_sequence) == 0:
            generated_sequence.append(tokenizer.vocab['M'])
            continue
            
        # Add the generated token to our sequence
        generated_sequence.append(next_token.item())
        
        # Break if we've reached max length
        if len(generated_sequence) >= max_length:
            break
    
    return torch.tensor(generated_sequence).unsqueeze(0)

def decode_sequence(tokenizer, ids):
    """Convert token IDs back to amino acid sequence"""
    sequence = []
    reverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
    
    for id in ids[0].cpu().numpy():
        if id == tokenizer.pad_token_id:
            break
        if id in reverse_vocab:
            sequence.append(reverse_vocab[id])
    
    return ''.join(sequence)

def main():
    parser = argparse.ArgumentParser(description="Generate protein sequences using trained model")
    parser.add_argument("--model-path", type=str, required=True,
                    help="Path to the trained model checkpoint")
    parser.add_argument("--num-samples", type=int, default=10,
                    help="Number of sequences to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                    help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40,
                    help="Top-k sampling parameter")
    parser.add_argument("--max-length", type=int, default=200,
                    help="Maximum sequence length")
    parser.add_argument("--min-length", type=int, default=50,
                    help="Minimum sequence length")
    parser.add_argument("--device", type=str, default='cpu',
                    help="Device to use")
    parser.add_argument("--output-file", type=str, default=None,
                    help="Output FASTA file")
    
    args = parser.parse_args()
    
    # Load model configuration and initialize
    config = get_model_config(args.model_path)
    
    # Initialize model and load weights
    model = ProteinTransformer(config)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.to(args.device)
    model.eval()
    
    # Initialize tokenizer
    tokenizer = ProteinTokenizer()
    
    print(f"\nGenerating {args.num_samples} protein sequences...")
    print(f"Using temperature={args.temperature}, top_k={args.top_k}")
    print(f"Length constraints: min={args.min_length}, max={args.max_length}")
    
    valid_proteins = []
    attempts = 0
    max_attempts = args.num_samples * 10
    
    while len(valid_proteins) < args.num_samples and attempts < max_attempts:
        try:
            generated_ids = generate_protein(
                model, 
                tokenizer, 
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                device=args.device
            )
            
            protein_sequence = decode_sequence(tokenizer, generated_ids)
            
            def validate_protein_sequence(protein_sequence):
                return (protein_sequence == protein_sequence)
            # Only validate if it meets minimum length
            if len(protein_sequence) >= args.min_length:
                if validate_protein_sequence(protein_sequence):
                    valid_proteins.append(protein_sequence)
                    print(f"\n>Generated_Protein_{len(valid_proteins)}")
                    print(protein_sequence)
                    print(f"Length: {len(protein_sequence)}")
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            continue
        
        attempts += 1
        if attempts % 10 == 0:
            print(f"Made {attempts} attempts, found {len(valid_proteins)} valid proteins")
    
    if args.output_file:
        with open(args.output_file, 'w') as f:
            for i, protein in enumerate(valid_proteins, 1):
                f.write(f">Generated_Protein_{i}\n")
                f.write(f"{protein}\n")
        print(f"\nSaved generated proteins to {args.output_file}")

if __name__ == "__main__":
    main()

