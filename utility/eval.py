import os
from pathlib import Path

import torch
from omegaconf import DictConfig

from midi_embeddings.train import evaluate
from midi_embeddings.visualize import visualize_embeddings
from midi_embeddings.transformer import MIDITransformerEncoder
from midi_embeddings.dataset import MIDIDatasetDynamic, MIDIDatasetPresaved


def evaluate_model(cfg: DictConfig, device: torch.device):
    """
    Model eval handler

    Args:
        cfg: Hydra config
        device: Device to use for evaluation (CPU or GPU)
    """
    # Set tokenizer path
    tokenizer_path = Path(cfg.dataset.tokenizer_path)
    if not tokenizer_path.exists():
        print(f"Tokenizer at {tokenizer_path} not found. Training required. Please wait.")
        tokenizer_path = None

    # Load visualization dataset
    viz_dataset = MIDIDatasetPresaved(
        split="all",
        max_seq_len=cfg.train.max_seq_len,
        tokenizer_path=tokenizer_path,
        return_info=True,
        limit=cfg.dataset.limit,
    )

    # If tokenizer was null, viz_dataset trained a new one
    tokenizer_path = viz_dataset.tokenizer_path

    # Load checkpoint
    checkpoint_path = cfg.eval.model_path
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    print(f"Loading model from {checkpoint_path} for evaluation...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Initialize model with the same configuration as the saved model
    if "config" in checkpoint:
        # Use config from checkpoint if available
        model_config = checkpoint["config"]
        model = MIDITransformerEncoder(
            vocab_size=viz_dataset.vocab_size,
            embed_dim=model_config["embed_dim"],
            nhead=model_config["nhead"],
            num_layers=model_config["num_layers"],
            max_seq_len=model_config["max_seq_len"],
            dropout=model_config["dropout"],
        ).to(device)
    else:
        # Use config from current config file
        model = MIDITransformerEncoder(
            vocab_size=viz_dataset.vocab_size,
            embed_dim=cfg.train.embed_dim,
            nhead=cfg.train.nhead,
            num_layers=cfg.train.num_layers,
            max_seq_len=cfg.train.max_seq_len,
            dropout=cfg.train.dropout,
        ).to(device)

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Evaluate the model on test dataset
    test_dataset = MIDIDatasetDynamic(
        split="test",
        max_seq_len=cfg.train.max_seq_len,
        tokenizer_path=tokenizer_path,
    )

    test_loss, perplexity = evaluate(model, test_dataset, device)
    print(f"Test Loss: {test_loss: .4f}, Perplexity: {perplexity: .4f}")

    # Visualize embeddings if requested
    if cfg.eval.visualize:
        print("Visualizing embeddings...")
        output_file = cfg.eval.output_file or "eval_embeddings.html"
        visualize_embeddings(
            model=model,
            device=device,
            max_seq_len=cfg.train.max_seq_len,
            file_name=output_file,
            dataset=viz_dataset,
        )
        print(f"Embeddings visualization saved to {output_file}")
