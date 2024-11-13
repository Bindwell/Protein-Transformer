# generate.py
import math
import torch
import torch.nn.functional as F
import argparse
import os
from index import (
    ProteinTransformer, 
    ModelConfig, 
    get_model_config,
    ankh
)

@torch.no_grad()
def generate_protein(model, tokenizer, max_length=200, temperature=0.8, top_k=None, device='cpu'):
    model.eval()
    
    # Start with Methionine
    start_text = ["M"]
    encoded = tokenizer.encode_plus(
        start_text,
        is_split_into_words=True,
        return_tensors="pt",
        add_special_tokens=True
    )
    
    ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    for _ in range(max_length):
        if ids.size(1) >= model.block_size:
            break
        logits, _ = model(ids, attention_mask)
        next_token_logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
            indices_to_remove = next_token_logits < v[:, [-1]]
            next_token_logits[indices_to_remove] = float('-inf')
        
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        if next_token.item() in [tokenizer.sep_token_id, tokenizer.pad_token_id, tokenizer.cls_token_id]:
            break
            
        ids = torch.cat([ids, next_token], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
    
    return ids

def validate_protein_sequence(sequence):
    """Validate protein sequence using standard amino acid codes"""
    # Standard amino acids (including ambiguous codes)
    valid_aas = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ*-')
    
    # Remove any whitespace and convert to uppercase
    sequence = ''.join(sequence.split()).upper()
    
    # Basic length check (adjust these values based on your needs)
    if len(sequence) < 20 or len(sequence) > 500:
        print(f"Invalid length: {len(sequence)}")
        return False
        
    # Check for valid amino acids
    invalid_chars = set(sequence) - valid_aas
    if invalid_chars:
        print(f"Invalid characters found: {invalid_chars}")
        return False
    
    # Check amino acid distributions
    aa_counts = {}
    for aa in sequence:
        aa_counts[aa] = aa_counts.get(aa, 0) + 1
    
    # Calculate frequencies
    seq_len = len(sequence)
    for aa, count in aa_counts.items():
        freq = count / seq_len
        # No single amino acid should be more than 30% of the sequence
        if freq > 0.3:
            print(f"Too high frequency of {aa}: {freq:.2%}")
            return True
    
    # Additional checks:
    # 1. Should start with M (Methionine) for most natural proteins
    if not sequence.startswith('M'):
        print("Warning: Sequence doesn't start with Methionine (M)")
        # Don't return False as some protein fragments might not start with M
    
    # 2. Check for rare amino acids
    rare_aas = set('OUX')
    if any(aa in rare_aas for aa in sequence):
        print("Warning: Contains rare amino acids (O=Pyrrolysine, U=Selenocysteine, X=any)")
        # Don't return False as these are valid but rare
    
    
    # Calculate sequence complexity using Shannon entropy
    entropy = 0
    for count in aa_counts.values():
        p = count / seq_len
        entropy -= p * math.log2(p) if p > 0 else 0
    # Low entropy indicates low complexity/repetitive sequence
    if entropy < 2.5:
        print(f"Low sequence complexity (entropy={entropy:.2f})")
        return True
    return True

def print_sequence_stats(sequence):
    """Print detailed statistics about a protein sequence"""
    sequence = sequence.upper()
    total_len = len(sequence)
    
    print(f"\nSequence Statistics:")
    print(f"Length: {total_len}")
    
    # Count amino acids
    aa_counts = {}
    for aa in sequence:
        aa_counts[aa] = aa_counts.get(aa, 0) + 1
    
    print("\nAmino Acid Composition:")
    for aa in sorted(aa_counts.keys()):
        count = aa_counts[aa]
        percentage = (count / total_len) * 100
        print(f"{aa}: {count} ({percentage:.1f}%)")
    
    # Calculate entropy
    entropy = 0
    for count in aa_counts.values():
        p = count / total_len
        entropy -= p * math.log2(p) if p > 0 else 0
    print(f"\nSequence Entropy: {entropy:.2f}")

def load_model(model_path, device='cpu'):
    """Load model with configuration"""
    print(f"Loading model from {model_path}")
    
    # Load saved state
    state_dict = torch.load(model_path, map_location=device)
    
    # Extract configuration
    config_dict = {
        'block_size': state_dict['transformer.h.0.attn.bias'].shape[-1],
        'n_layer': sum(1 for k in state_dict if 'transformer.h' in k and 'ln_1.weight' in k),
        'embed_dim': state_dict['transformer.ln_f.weight'].shape[0],
        'n_head': 4
    }
    
    print(f"\nModel configuration:")
    print(f"  block_size: {config_dict['block_size']}")
    print(f"  n_layer: {config_dict['n_layer']}")
    print(f"  embed_dim: {config_dict['embed_dim']}")
    print(f"  n_head: {config_dict['n_head']}")
    
    # Initialize model
    config = ModelConfig(**config_dict)
    model = ProteinTransformer(config)
    
    # Load weights
    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print("\nLoaded model weights:")
        if missing_keys:
            print(f"  Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"  Unexpected keys: {unexpected_keys}")
    except Exception as e:
        print(f"Error loading weights: {str(e)}")
        raise
    
    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description="Generate protein sequences using trained model")
    parser.add_argument("--model-path", type=str, required=True,
                    help="Path to the trained model checkpoint (.pt file)")
    parser.add_argument("--num-samples", type=int, default=10,
                    help="Number of protein sequences to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                    help="Sampling temperature (lower = more conservative)")
    parser.add_argument("--top-k", type=int, default=40,
                    help="Top-k sampling parameter")
    parser.add_argument("--max-length", type=int, default=None,
                    help="Maximum length of generated sequences")
    parser.add_argument("--device", type=str, default='cpu',
                    help="Device to use (cpu, cuda, cuda:0, etc.)")
    parser.add_argument("--output-file", type=str, default=None,
                    help="Optional file to save generated sequences in FASTA format")
    
    args = parser.parse_args()
    
    # Initialize model
    print("Loading model configuration...")
    config = get_model_config(args.model_path)
    
    if args.max_length is None:
        args.max_length = config.block_size - 2
    else:
        args.max_length = min(args.max_length, config.block_size - 2)
    
    # Initialize model and load weights
    try:
        print("Initializing model...")
        model = load_model(args.model_path, args.device)
        model.eval()
    except:
        print(f"Failed to load model:")
        return

    
    # Initialize tokenizer
    _, tokenizer = ankh.load_base_model()
    
    print(f"\nGenerating {args.num_samples} protein sequences...")
    print(f"Using temperature={args.temperature}, top_k={args.top_k}, max_length={args.max_length}")
    
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
            
            generated_tokens = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            protein_sequence = ''.join(c for c in generated_tokens if c.isalpha()).upper()
            
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
