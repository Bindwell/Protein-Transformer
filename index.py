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
from einops import rearrange
import ankh

@dataclass
class ModelConfig:
    block_size: int = None  # length of the input sequences
    n_layer: int = 4
    embed_dim: int = 768  # Match Ankh's embedding dimension
    n_head: int = 4

# Add this after the ModelConfig class and before the AnkhProteinDataset class:

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """
    def __init__(self, config):
        super().__init__()
        assert config.embed_dim % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.embed_dim, 3 * config.embed_dim)
        # output projection
        self.c_proj = nn.Linear(config.embed_dim, config.embed_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.embed_dim = config.embed_dim

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (embed_dim)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.embed_dim, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = rearrange(y, "b h t c -> b t (h c)")
        # output projection
        y = self.c_proj(y)
        return y

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

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
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


class AnkhProteinDataset(Dataset):
    """Dataset for protein sequences using Ankh tokenization"""

    def __init__(self, proteins, max_protein_length):
        self.proteins = proteins
        self.max_protein_length = max_protein_length
        _, self.tokenizer = ankh.load_base_model()
        
    def __len__(self):
        return len(self.proteins)

    def contains(self, word):
        return word in self.proteins

    def get_output_length(self):
        return self.max_protein_length

    def __getitem__(self, idx):
        protein = self.proteins[idx]
        protein_chars = list(protein)
        
        # Tokenize the protein sequence
        outputs = self.tokenizer.encode_plus(
            protein_chars,
            add_special_tokens=True,
            padding='max_length',
            max_length=self.max_protein_length,
            truncation=True,
            is_split_into_words=True,
            return_tensors="pt"
        )
        
        x = outputs['input_ids'].squeeze(0)
        attention_mask = outputs['attention_mask'].squeeze(0)
        
        # Create target sequence (shifted by 1 position)
        y = torch.roll(x, -1)
        y[-1] = -1  # Mark last position as padding
        y = y.masked_fill(attention_mask == 0, -1)  # Mask padding positions
        
        return x, y, attention_mask


class ProteinTransformer(nn.Module):
    """Modified Transformer that uses Ankh embeddings"""
    
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        
        # Load Ankh model for embeddings
        self.ankh_model, _ = ankh.load_base_model()
        self.ankh_model.eval()  # Set to eval mode as we'll only use it for embeddings
        
        # Freeze Ankh parameters
        for param in self.ankh_model.parameters():
            param.requires_grad = False
            
        self.transformer = nn.ModuleDict(dict(
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.embed_dim),
        ))
        
        self.lm_head = nn.Linear(config.embed_dim, self.ankh_model.config.vocab_size, bias=False)
        
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of trainable parameters: %.2fM" % (n_params/1e6,))

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, attention_mask, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"

        # Get embeddings from Ankh model
        with torch.no_grad():
            ankh_outputs = self.ankh_model(
                input_ids=idx,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            x = ankh_outputs.hidden_states[-1]  # Use the last hidden layer

        # Forward through our transformer layers
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                                 targets.view(-1), 
                                 ignore_index=-1)

        return logits, loss

def create_datasets(input_file):
    # Read proteins from FASTA file
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

    # Create datasets using Ankh tokenization
    train_dataset = AnkhProteinDataset(train_proteins, max_word_length)
    test_dataset = AnkhProteinDataset(test_proteins, max_word_length)

    return train_dataset, test_dataset

def save_model_state(model, path):
    """Save only the trainable parts of the model"""
    # Get state dict
    state_dict = model.state_dict()
    
    # Filter out ankh_model parameters
    save_dict = {k: v for k, v in state_dict.items() 
                if not k.startswith('ankh_model.')}
    
    # Save configuration along with weights
    config_dict = {
        'block_size': model.block_size,
        'n_layer': len(model.transformer.h),
        'embed_dim': model.transformer.ln_f.weight.shape[0],
        'n_head': model.transformer.h[0].attn.n_head if model.transformer.h else 4
    }
    
    # Save both weights and config
    torch.save({
        'config': config_dict,
        'model_state': save_dict
    }, path)



# Modified training loop
def train_model(model, train_dataset, test_dataset, args):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True
    )
    for step, batch in enumerate(train_loader):
        x, y, attention_mask = [t.to(args.device) for t in batch]
        # Forward pass
        logits, loss = model(x, attention_mask, y)
        # Backward pass
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
        if args.max_steps >= 0 and step >= args.max_steps:
            break

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
        n_head=4
    )
    print(f"Extracted config from model:")
    print(f"  block_size: {config.block_size}")
    print(f"  n_layer: {config.n_layer}")
    print(f"  embed_dim: {config.embed_dim}")
    print(f"  n_head: {config.n_head}")
    return config

if __name__ == '__main__':
    # parse command line args
    parser = argparse.ArgumentParser(description="Make more proteins using Ankh embeddings")
    parser.add_argument('--input-file', '-i', type=str, default='data/hypf.fa', help="input fasta file")
    parser.add_argument('--work-dir', '-o', type=str, default='out', help="output working directory")
    parser.add_argument('--resume', action='store_true', help="when this flag is used, we will resume optimization from existing model in the workdir")
    parser.add_argument('--sample-only', action='store_true', help="just sample from the model and quit, don't train")
    parser.add_argument('--max-steps', type=int, default=75_000, help="max number of optimization steps to run for, or -1 for infinite")
    parser.add_argument('--device', type=str, default='cpu', help="device to use for compute, examples: cpu|cuda|cuda:2|mps")
    parser.add_argument('--seed', type=int, default=42, help="seed")
    parser.add_argument('--n-layer', type=int, default=4, help="number of layers")
    parser.add_argument('--n-head', type=int, default=4, help="number of heads")
    parser.add_argument('--batch-size', '-b', type=int, default=32, help="batch size during optimization")
    parser.add_argument('--learning-rate', '-l', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--weight-decay', '-w', type=float, default=0.01, help="weight decay")
    args = parser.parse_args()
    print(vars(args))

    # system inits
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.work_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.work_dir)

    # init datasets
    print("Initializing datasets...")
    train_dataset, test_dataset = create_datasets(args.input_file)
    block_size = train_dataset.get_output_length()
    print(f"Maximum sequence length: {block_size}")

    # init model
    print("Initializing model...")
    config = ModelConfig(
        block_size=block_size,
        n_layer=args.n_layer,
        n_head=args.n_head
    )
    model = ProteinTransformer(config)
    model.to(args.device)
    if args.resume or args.sample_only:
        print("Resuming from existing model in the workdir")
        model.load_state_dict(torch.load(os.path.join(args.work_dir, 'model.pt'), 
                                       map_location=torch.device(args.device)))
    if args.sample_only:
        print("Sampling mode - generating proteins...")
        # Implement sampling logic here if needed
        sys.exit()

    # Initialize optimizer
    print("Initializing optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True
    )

    # Training loop
    print("Starting training...")
    best_loss = None
    step = 0
    
    try:
        for epoch in range(args.max_steps):
            for batch in train_loader:
                t0 = time.time()

                # Move batch to device
                x, y, attention_mask = [t.to(args.device) for t in batch]

                # Forward pass
                logits, loss = model(x, attention_mask, y)

                # Backward pass and optimize
                model.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                # Timing
                if args.device.startswith('cuda'):
                    torch.cuda.synchronize()
                t1 = time.time()

                # Logging
                if step % 10 == 0:
                    print(f"step {step} | loss {loss.item():.4f} | step time {(t1-t0)*1000:.2f}ms")

                # Evaluation
                if step > 0 and step % 100 == 0:
                    model.eval()
                    with torch.no_grad():
                        # Evaluate on train set
                        train_losses = []
                        for eval_batch in DataLoader(train_dataset, batch_size=100):
                            eval_x, eval_y, eval_mask = [t.to(args.device) for t in eval_batch]
                            _, train_loss = model(eval_x, eval_mask, eval_y)
                            train_losses.append(train_loss.item())
                        avg_train_loss = sum(train_losses) / len(train_losses)

                        # Evaluate on test set
                        test_losses = []
                        for eval_batch in DataLoader(test_dataset, batch_size=100):
                            eval_x, eval_y, eval_mask = [t.to(args.device) for t in eval_batch]
                            _, test_loss = model(eval_x, eval_mask, eval_y)
                            test_losses.append(test_loss.item())
                        avg_test_loss = sum(test_losses) / len(test_losses)

                    writer.add_scalar("Loss/train", avg_train_loss, step)
                    writer.add_scalar("Loss/test", avg_test_loss, step)
                    writer.flush()
                    print(f"step {step} train loss: {avg_train_loss:.4f} test loss: {avg_test_loss:.4f}")

                    # Save best model
                    if best_loss is None or avg_test_loss < best_loss:
                        out_path = os.path.join(args.work_dir, "model.pt")
                        print(f"Test loss {avg_test_loss:.4f} is the best so far, saving model to {out_path}")
                        save_model_state(model, args.work_dir)
                        best_loss = avg_test_loss
                    
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
        # Final save
        out_path = os.path.join(args.work_dir, "model_final.pt")
        print(f"Saving final model to {out_path}")
        torch.save(model.state_dict(), out_path)
        writer.close()

    print("Training completed")