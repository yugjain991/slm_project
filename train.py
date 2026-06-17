# train.py — Word-Level Training Script for GPT Language Model
# ============================================================
# Key change: uses WORD-LEVEL tokenization instead of character-level.
# Tokenizes text into words and punctuation using regex, builds a vocabulary,
# and trains the GPT model with cosine annealing LR schedule.

import torch
import torch.nn as nn
import pickle
import re
import os

from model import GPTLanguageModel

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
BLOCK_SIZE = 64      # Maximum context length (in tokens/words)
BATCH_SIZE = 8       # Number of sequences per training batch
MAX_STEPS = 10000    # Total training iterations
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
EVAL_INTERVAL = 200  # Print loss every N steps
SAMPLE_INTERVAL = 1000  # Generate sample text every N steps
SAMPLE_LENGTH = 30   # Number of tokens to generate in samples

# Model hyperparameters
N_EMBD = 128
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.1

# Device selection: use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# -------------------------------------------------------------------
# Step 1: Load the training data
# -------------------------------------------------------------------
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stories.txt')
print(f"Loading data from: {data_path}")

with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Total characters in dataset: {len(text):,}")

# -------------------------------------------------------------------
# Step 2: Build word-level vocabulary
# -------------------------------------------------------------------
# Lowercase all text for consistency
text = text.lower()

# Tokenize: split into words and individual punctuation characters
# This regex captures sequences of alphanumeric chars OR single non-space chars
tokens = re.findall(r"[a-zA-Z0-9]+|[^\s]", text)
print(f"Total tokens (words + punctuation): {len(tokens):,}")

# Build vocabulary from unique tokens, sorted for reproducibility
unique_tokens = sorted(set(tokens))

# Special tokens go at the start of the vocabulary
# <PAD>=0 for padding, <UNK>=1 for unknown words, <END>=2 for end-of-text
special_tokens = ['<PAD>', '<UNK>', '<END>']
vocab = special_tokens + unique_tokens

vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size:,}")

# Create string-to-index and index-to-string mappings
stoi = {word: i for i, word in enumerate(vocab)}
itos = {i: word for i, word in enumerate(vocab)}

# Save vocabulary for use during inference/generation
vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vocab.pkl')
with open(vocab_path, 'wb') as f:
    pickle.dump({'stoi': stoi, 'itos': itos, 'vocab': vocab}, f)
print(f"Vocabulary saved to: {vocab_path}")


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------
def encode(token_list):
    """Convert a list of word tokens to a list of integer indices."""
    return [stoi.get(t, stoi['<UNK>']) for t in token_list]


def decode(indices):
    """Convert a list of integer indices back to readable text.

    Joins tokens with spaces — since we tokenize words and punctuation
    separately, punctuation will have spaces around it. This is acceptable
    for a word-level model.
    """
    return ' '.join([itos.get(i, '<UNK>') for i in indices])


# -------------------------------------------------------------------
# Step 3: Encode the full text as token indices
# -------------------------------------------------------------------
data = torch.tensor(encode(tokens), dtype=torch.long)
print(f"Encoded data shape: {data.shape}")
print(f"Sample encoding (first 20 tokens): {[itos[i.item()] for i in data[:20]]}")


# -------------------------------------------------------------------
# Step 4: Batch generation
# -------------------------------------------------------------------
def get_batch():
    """Generate a random batch of training examples.

    For each example in the batch:
    - Pick a random starting position in the data
    - x = tokens[start : start + block_size]       (input)
    - y = tokens[start+1 : start + block_size + 1] (target, shifted by 1)

    Returns:
        x: (batch_size, block_size) input tensor
        y: (batch_size, block_size) target tensor
    """
    # Random start positions (ensure we don't go past the end)
    ix = torch.randint(len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x.to(device), y.to(device)


# -------------------------------------------------------------------
# Step 5: Initialize model, optimizer, and scheduler
# -------------------------------------------------------------------
model = GPTLanguageModel(
    vocab_size=vocab_size,
    n_embd=N_EMBD,
    n_head=N_HEAD,
    n_layer=N_LAYER,
    block_size=BLOCK_SIZE,
    dropout=DROPOUT,
).to(device)

total_params = model.count_parameters()
print(f"\nModel initialized with {total_params:,} trainable parameters")
print(f"Architecture: {N_LAYER} layers, {N_HEAD} heads, {N_EMBD} embedding dim")
print(f"Block size: {BLOCK_SIZE}, Dropout: {DROPOUT}\n")

# AdamW optimizer with weight decay for regularization
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

# Cosine annealing: smoothly reduces LR from initial to ~0 over training
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=MAX_STEPS,
)

# -------------------------------------------------------------------
# Step 6: Training loop
# -------------------------------------------------------------------
print("=" * 60)
print("Starting training...")
print("=" * 60)

best_loss = float('inf')

for step in range(1, MAX_STEPS + 1):
    model.train()

    # Get a batch of training data
    xb, yb = get_batch()

    # Forward pass
    logits = model(xb)  # (B, T, vocab_size)

    # Compute cross-entropy loss
    # Reshape logits and targets for nn.functional.cross_entropy
    B, T, C = logits.shape
    logits = logits.view(B * T, C)
    targets = yb.view(B * T)
    loss = nn.functional.cross_entropy(logits, targets)

    # Backward pass and optimization
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    # Gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    # Track best loss
    current_loss = loss.item()
    if current_loss < best_loss:
        best_loss = current_loss

    # Print progress
    if step % EVAL_INTERVAL == 0:
        current_lr = scheduler.get_last_lr()[0]
        print(f"Step {step:>5d}/{MAX_STEPS} | "
              f"Loss: {current_loss:.4f} | "
              f"Best: {best_loss:.4f} | "
              f"LR: {current_lr:.2e}")

    # Generate a sample to show training progress
    if step % SAMPLE_INTERVAL == 0:
        model.eval()
        # Start generation from a random token
        start_idx = torch.randint(vocab_size, (1, 1)).to(device)
        generated = model.generate(start_idx, max_new_tokens=SAMPLE_LENGTH)
        generated_text = decode(generated[0].tolist())
        print(f"\n--- Sample at step {step} ---")
        print(generated_text)
        print("---\n")

# -------------------------------------------------------------------
# Step 7: Save the trained model
# -------------------------------------------------------------------
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pth')
torch.save(model.state_dict(), model_path)
print(f"\nModel saved to: {model_path}")

# -------------------------------------------------------------------
# Final summary
# -------------------------------------------------------------------
print("\n" + "=" * 60)
print("Training complete!")
print("=" * 60)
print(f"Total trainable parameters: {total_params:,}")
print(f"Final loss: {current_loss:.4f}")
print(f"Best loss:  {best_loss:.4f}")
print(f"Vocabulary size: {vocab_size:,}")
print(f"Model saved to: {model_path}")
print(f"Vocab saved to: {vocab_path}")