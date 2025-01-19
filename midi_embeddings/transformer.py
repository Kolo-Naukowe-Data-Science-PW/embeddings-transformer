import torch.nn as nn
import torch

class MIDITransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, num_layers, max_seq_len, dropout=0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, 
                                                   nhead=nhead, 
                                                   batch_first=True,
                                                   dropout=dropout)

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.output_layer = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, tokens, src_mask=None):
        seq_len = tokens.size(1)
        positions = torch.arange(0, seq_len, device=tokens.device).unsqueeze(0)
        
        # Łączenie embeddingów
        x = (self.token_embedding(tokens) +
             self.position_embedding(positions))
        
        # Maskowane przetwarzanie w enkoderze
        x = self.encoder(x, mask=src_mask)  # [batch_size, seq_len, embed_dim]
        
        # Predykcja pitch, velocity i time
        return self.output_layer(x)
    
    def get_embeddings(self, tokens):
        """Get song embedding by applying mean pooling to the transformer hidden states"""
        tokens = tokens.unsqueeze(0)  # Add batch dimension (batch_size=1)
        
        positions = torch.arange(0, tokens.size(1), device=tokens.device).unsqueeze(0)
        # Pass tokens through the model to get hidden states (before the output layer)
        with torch.no_grad():
            hidden_states = self.encoder(self.token_embedding(tokens) + self.position_embedding(positions))  # Shape: [1, seq_len, embed_dim]
        
        # Mean pooling across sequence length (dim=1) to get a fixed-size embedding
        song_embedding = hidden_states.mean(dim=1)  # Shape: [1, embed_dim]
        
        return song_embedding


def generate_causal_mask(seq_len, device):
    # Generowanie maski dla przewidywania następnego tokena
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask