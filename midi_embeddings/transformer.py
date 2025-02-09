"""
Transformer Model Module (transformer.py)
-----------------------------------------
This module implements the Transformer model for processing MIDI data.
It includes:

- `MIDITransformerEncoder`: A transformer-based encoder for MIDI sequences.
- `generate_causal_mask`: Generates a causal mask for autoregressive training.
"""

import torch
import torch.nn as nn


class MIDITransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        nhead: int,
        num_layers: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        """Transformer model for MIDI data

        Args:
            vocab_size (int): size of the vocabulary
            embed_dim (int): size of embedding dimensions
            nhead (int): number of model's attention heads
            num_layers (int): number of layers of the model
            max_seq_len (int): maximum length of processed sequence
            dropout (float, optional): dropout rate. Defaults to 0.1.
        """
        super().__init__()

        # Initialize embedding modules (token + position)
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        # Initialize encoder's layer and the encoder itself
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            batch_first=True,
            dropout=dropout,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, tokens: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """Pass the tokens through the model and return the predictions

        Args:
            tokens (tensor): tokens tensor. Shape: [batch_size, seq_len]
            src_mask (tensor, optional): Triangular mask for causal training

        Returns:
            tensor: Model predictions for current tokens input.
        """
        # Tokens: [batch_size, seq_len]
        seq_len = tokens.size(1)  # choose seq_len only

        # Create positions tensor: [batch_size, seq_len] (0, 1, 2, ..., seq_len-1)
        positions = torch.arange(0, seq_len, device=tokens.device).unsqueeze(0)

        # Combine both embeddings
        x = self.token_embedding(tokens) + self.position_embedding(positions)

        # Masked pass through the model
        x = self.encoder(x, mask=src_mask)  # [batch_size, seq_len, embed_dim]

        # Return the predictions
        return self.output_layer(x)

    def get_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        """Get song embedding by applying mean pooling to the transformer hidden states

        Args:
            tokens (tensor): Input song tokens.

        Returns:
            tensor: Song embedding.
        """
        tokens = tokens.unsqueeze(0)  # Add batch dimension (batch_size=1)

        # Create positions tensor: [batch_size, seq_len] (0, 1, 2, ..., seq_len-1)
        positions = torch.arange(0, tokens.size(1), device=tokens.device).unsqueeze(0)

        # Pass tokens through the model to get hidden states (without the output layer)
        with torch.no_grad():
            hidden_states = self.encoder(
                self.token_embedding(tokens) + self.position_embedding(positions)
            )  # Shape: [1, seq_len, embed_dim]

        # Mean pooling across sequence length (dim=1) to get a fixed-size embedding
        song_embedding = hidden_states.mean(dim=1)  # Shape: [1, embed_dim]

        return song_embedding


def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Generates the triangular mask for causal training. Prevents the model from
    attending to future tokens.

    Args:
        seq_len (int): Length of the sequence.
        device (torch.device): Device to create the mask on.

    Returns:
        torch.Tensor: Causal mask tensor.
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float("-inf"))

    return mask
