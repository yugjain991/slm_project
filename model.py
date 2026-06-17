# model.py — GPT-style Language Model (Word-Level)
# ================================================
# A scaled-up transformer language model designed for word-level tokenization.
# Uses pre-norm architecture, GELU activations, and proper weight initialization.

import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """Single head of self-attention."""

    def __init__(self, head_size, n_embd, block_size, dropout=0.1):
        super().__init__()
        # Linear projections for key, query, value (no bias for efficiency)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Causal mask: lower-triangular matrix to prevent attending to future tokens
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (B, T, C) where B=batch, T=sequence length, C=n_embd
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        # Scaled dot-product attention
        # Scale by 1/sqrt(head_size) to keep variance stable
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)  # (B, T, T)
        # Apply causal mask: set future positions to -inf so softmax gives them 0
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        # Weighted aggregation of values
        out = wei @ v  # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention running in parallel."""

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout=0.1):
        super().__init__()
        # Create multiple attention heads
        self.heads = nn.ModuleList([
            Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)
        ])
        # Projection layer to mix head outputs back to n_embd dimensions
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Run all heads in parallel, concatenate their outputs
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, head_size * num_heads)
        # Project back to n_embd and apply dropout
        out = self.dropout(self.proj(out))  # (B, T, n_embd)
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation."""

    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        # Expand to 4x, apply GELU, project back down
        # GELU is smoother than ReLU and works better for language models
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: pre-norm architecture with residual connections.

    Pre-norm applies LayerNorm BEFORE attention/ffwd (more stable training
    compared to post-norm which applies LayerNorm after).
    """

    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        head_size = n_embd // n_head
        # Layer norms applied before attention and feed-forward (pre-norm)
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout)

    def forward(self, x):
        # Residual connections: add input back to output of each sub-layer
        x = x + self.sa(self.ln1(x))    # Self-attention with residual
        x = x + self.ffwd(self.ln2(x))  # Feed-forward with residual
        return x


class GPTLanguageModel(nn.Module):
    """GPT-style autoregressive language model for word-level generation.

    Architecture:
        Token embeddings + positional embeddings
        → N transformer blocks (pre-norm)
        → Final LayerNorm
        → Linear head to vocab logits
    """

    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=4,
                 block_size=64, dropout=0.1):
        super().__init__()
        # Store all hyperparameters as instance attributes
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.dropout = dropout

        # Token and position embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # Stack of transformer blocks
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)
        ])

        # Final layer norm (applied after all transformer blocks)
        self.ln_f = nn.LayerNorm(n_embd)

        # Language model head: project from embedding dim to vocabulary
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Initialize weights for better training stability
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Custom weight initialization following GPT-2 conventions.

        - Linear layers and Embeddings: normal distribution with std=0.02
        - LayerNorm: weight=1, bias=0
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, idx):
        """Forward pass through the model.

        Args:
            idx: (B, T) tensor of token indices

        Returns:
            logits: (B, T, vocab_size) tensor of unnormalized predictions
        """
        B, T = idx.shape
        device = idx.device

        # Look up token embeddings and add positional information
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)          # (T,)
        )                                            # (T, n_embd)
        x = tok_emb + pos_emb                       # (B, T, n_embd) — broadcast addition

        # Pass through transformer blocks and final norm
        x = self.blocks(x)     # (B, T, n_embd)
        x = self.ln_f(x)       # (B, T, n_embd)

        # Project to vocabulary logits
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.7):
        """Autoregressively generate new tokens.

        Args:
            idx: (B, T) tensor of starting token indices
            max_new_tokens: number of new tokens to generate
            temperature: controls randomness (lower = more deterministic,
                         higher = more creative). Default 0.7 is a good balance.

        Returns:
            idx: (B, T + max_new_tokens) tensor with generated tokens appended
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size (model's maximum context window)
            idx_cond = idx[:, -self.block_size:]

            # Get predictions
            logits = self(idx_cond)  # (B, T, vocab_size)

            # Focus on the last time step only (next-token prediction)
            logits = logits[:, -1, :]  # (B, vocab_size)

            # Apply temperature scaling before softmax
            logits = logits / temperature

            # Convert to probabilities and sample
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append sampled token to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx

    def count_parameters(self):
        """Returns the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)