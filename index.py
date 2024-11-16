import os
import sys
import time
import math
import argparse
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from biotite.sequence.io.fasta import FastaFile
import ankh as ankh
from einops import rearrange

@dataclass
class ModelConfig:
    block_size: int = None  # length of the input sequences
    vocab_size: int = 22  # 21 amino acids + padding token
    n_layer: int = 4
    embed_dim: int = 768
    n_head: int = 4

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


class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
class ProteinTokenizer:
    def __init__(self):
        self.vocab = {
            'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
            'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
            'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
            'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
            'U': 21, 'O': 21, # Unusual amino acids
            'X': 21, # Unknown or any amino acid
            'Z': 21, # Glutamic acid or glutamine
            'B': 21, # Asparagine or aspartic acid
            'J': 21, # Leucine or isoleucine
        }
        self.pad_token_id = 0
        
    def __call__(self, sequences, max_len=None, padding=True, truncation=True):
        tokenized = []
        for seq in sequences:
            tokens = [self.vocab.get(aa, self.vocab['X']) for aa in seq] # 'X' for unknown amino acids
            seq_len = len(tokens)
            if truncation and max_len is not None and seq_len > max_len:
                tokens = tokens[:max_len]
            if padding and max_len is not None:
                tokens += [self.pad_token_id] * (max_len - len(tokens))
            tokenized.append(tokens)
        return torch.tensor(tokenized)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.embed_dim % config.n_head == 0
        self.c_attn = nn.Linear(config.embed_dim, 3 * config.embed_dim)
        self.c_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.embed_dim = config.embed_dim

    def forward(self, x, attention_mask=None):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.embed_dim, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        if attention_mask is not None:
            att = att.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = rearrange(y, "b h t c -> b t (h c)")
        y = self.c_proj(y)
        return y

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.embed_dim)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.embed_dim)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.embed_dim, 4 * config.embed_dim),
            c_proj  = nn.Linear(4 * config.embed_dim, config.embed_dim),
            act     = NewGELU(),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x)))

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln_1(x), attention_mask)
        x = x + self.mlpf(self.ln_2(x))
        return x

class ProteinDataset(Dataset):
    def __init__(self, proteins, max_protein_length):
        self.proteins = proteins
        self.max_protein_length = max_protein_length
        self.tokenizer = ProteinTokenizer()
        
    def __len__(self):
        return len(self.proteins)

    def contains(self, word):
        return word in self.proteins

    def get_output_length(self):
        return self.max_protein_length

    def decode(self, ids):
        tokens = []
        for id in ids:
            if id == 0:  # pad token
                break
            for amino, idx in self.tokenizer.vocab.items():
                if idx == id:
                    tokens.append(amino)
                    break
        return ''.join(tokens)

    def __getitem__(self, idx):
        protein = self.proteins[idx]
        x = self.tokenizer([protein], max_len=self.max_protein_length)[0]
        attention_mask = (x != self.tokenizer.pad_token_id).float()
        
        # Create target sequence (shifted by 1 position)
        y = torch.roll(x, -1)
        y[-1] = -1  # Mark last position as padding
        y = y.masked_fill(attention_mask == 0, -1)  # Mask padding positions
        
        return x, y, attention_mask

class ProteinTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.embed_dim),
            wpe = nn.Embedding(config.block_size, config.embed_dim),
            drop = nn.Dropout(0.1),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.embed_dim),
        ))
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        
        n_params = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, attention_mask=None, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        
        # Get positions
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        
        # Forward the model
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for block in self.transformer.h:
            x = block(x, attention_mask)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                                 targets.view(-1), 
                                 ignore_index=-1)

        return logits, loss

@torch.no_grad()
def generate(model, tokenizer, start_sequence=None, max_new_tokens=100, temperature=1.0, top_k=None):
    model.eval()
    
    if start_sequence is None:
        x = torch.zeros((1, 1), dtype=torch.long, device=next(model.parameters()).device)
    else:
        tokens = tokenizer([start_sequence], max_len=model.get_block_size())[0]
        x = tokens.unsqueeze(0).to(next(model.parameters()).device)
    
    for _ in range(max_new_tokens):
        # Crop sequence if it's too long
        if x.size(1) > model.get_block_size():
            x = x[:, -model.get_block_size():]
            
        # Forward the model
        logits, _ = model(x)
        logits = logits[:, -1, :] / temperature
        
        # Optional top-k sampling
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
        
        # Sample from the distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append sampled token
        x = torch.cat((x, next_token), dim=1)
        
        # Stop if we sample the padding token
        if next_token.item() == tokenizer.pad_token_id:
            break
    
    return x

def print_samples(model, train_dataset, num=2, temperature=0.8, top_k=40):
    generated = []
    for _ in range(num):
        tokens = generate(model, train_dataset.tokenizer, max_new_tokens=model.get_block_size(), 
                        temperature=temperature, top_k=top_k)[0].tolist()
        protein = train_dataset.decode(tokens)
        generated.append(protein)
        
    print(f"\nGenerated {num} samples:")
    for i, protein in enumerate(generated, 1):
        is_train = train_dataset.contains(protein)
        status = "TRAIN" if is_train else "NOVEL"
        print(f">{i} {status}")
        if protein:  # Only print if protein is not empty
            print(protein)

def create_datasets(input_file):
    proteins = []
    fasta_file = FastaFile.read(input_file)
    for header, sequence in fasta_file.items():
        proteins.append(sequence)
    max_word_length = max(len(w) for w in proteins)

    # Split into train and test sets
    test_set_size = int(len(proteins) * 0.1)
    rp = torch.randperm(len(proteins)).tolist()
    train_proteins = [proteins[i] for i in rp[:-test_set_size]]
    test_proteins = [proteins[i] for i in rp[-test_set_size:]]
    
    print(f"split up the dataset into {len(train_proteins)} training examples and {len(test_proteins)} test examples")
    print(f"max protein length: {max_word_length}")

    train_dataset = ProteinDataset(train_proteins, max_word_length)
    test_dataset = ProteinDataset(test_proteins, max_word_length)

    return train_dataset, test_dataset

@torch.inference_mode()
def evaluate(model, dataset, batch_size=50, max_batches=None):
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y, mask = [t.to(next(model.parameters()).device) for t in batch]
        _, loss = model(x, mask, y)
        losses.append(loss.item())
    model.train()
    return torch.tensor(losses).mean().item()

def train():
    parser = argparse.ArgumentParser(description="Train protein generator")
    parser.add_argument('--input-file', '-i', type=str, required=True)
    parser.add_argument('--work-dir', '-o', type=str, default='out')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--sample-only', action='store_true')
    parser.add_argument('--max-steps', type=int, default=75_000)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-layer', type=int, default=4)
    parser.add_argument('--n-head', type=int, default=4)
    parser.add_argument('--embed-dim', type=int, default=768)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    os.makedirs(args.work_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.work_dir)
    device = torch.device(args.device)

    # Data
    train_dataset, test_dataset = create_datasets(args.input_file)
    
    # Model
    config = ModelConfig(
        block_size=train_dataset.get_output_length(),
        vocab_size=22,
        n_layer=args.n_layer,
        embed_dim=args.embed_dim,
        n_head=args.n_head
    )
    model = ProteinTransformer(config).to(device)
    
    if args.resume or args.sample_only:
        model.load_state_dict(torch.load(os.path.join(args.work_dir, 'model.pt')))
        if args.sample_only:
            print_samples(model, train_dataset, num=50)
            return

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    best_loss = float('inf')
    step = 0

    try:
        for epoch in range(1000000):
            for batch in train_loader:
                t0 = time.time()
                
                # Training step
                x, y, mask = [t.to(device) for t in batch]
                logits, loss = model(x, mask, y)
                model.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                
                # Logging
                if step % 10 == 0:  # Changed from 10 to 100 for less frequent logging
                    print(f"step {step} | loss {loss.item():.4f} | time {(time.time()-t0)*1000:.2f}ms")

                # Evaluation and Checkpointing
                if step > 0 and step % 1000 == 0:  # Changed from 500 to 1000 for less frequent evaluation
                    model.eval()
                    with torch.no_grad():
                        train_loss = evaluate(model, train_dataset)
                        test_loss = evaluate(model, test_dataset)
                        writer.add_scalar("Loss/train", train_loss, step)
                        writer.add_scalar("Loss/test", test_loss, step)
                        print(f"step {step} train loss: {train_loss:.4f} test loss: {test_loss:.4f}")
                        
                        if test_loss < best_loss:
                            best_loss = test_loss
                            torch.save(model.state_dict(), os.path.join(args.work_dir, 'model.pt'))
                            print(f"Saved new best model with test loss {test_loss:.4f}")
                            
                            # Only generate samples when we save a new best model
                            print_samples(model, train_dataset, num=5)
                    
                    model.train()
                
                step += 1
                if args.max_steps >= 0 and step >= args.max_steps:
                    print("Reached maximum steps, stopping training")
                    break
            
            if args.max_steps >= 0 and step >= args.max_steps:
                break
    
    except KeyboardInterrupt:
        print("Training interrupted by user")
    
    finally:
        # Final save and cleanup
        final_path = os.path.join(args.work_dir, 'model_final.pt')
        torch.save(model.state_dict(), final_path)
        print(f"Saved final model to {final_path}")
        writer.close()
        
    print("Training completed")

if __name__ == '__main__':
    train()
