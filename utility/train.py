from pathlib import Path

import torch
from omegaconf import DictConfig

from midi_embeddings.train import evaluate, train_model
from midi_embeddings.visualize import visualize_embeddings
from midi_embeddings.transformer import MIDITransformerEncoder
from midi_embeddings.dataset import MIDIDatasetDynamic, MIDIDatasetPresaved


def train(cfg: DictConfig, device: torch.device):
    """
    Model training handler

    Args:
        cfg: Hydra configuration
        device: Device for training (CPU/GPU)
    """
    # Set tokenizer path
    tokenizer_path = Path(cfg.dataset.tokenizer_path)
    if not tokenizer_path.exists():
        print(f"Tokenizer at {tokenizer_path} not found. Training required. Please wait.")
        tokenizer_path = None

    # Load dataset for visualization
    viz_dataset = MIDIDatasetPresaved(
        split="all",
        max_seq_len=cfg.train.max_seq_len,
        tokenizer_path=tokenizer_path,
        return_info=True,
        limit=cfg.dataset.limit,
    )

    # If tokenizer was null, viz_dataset trained a new one
    tokenizer_path = viz_dataset.tokenizer_path

    # Load datasets for training and validation
    train_dataset = MIDIDatasetPresaved(
        split="train",
        max_seq_len=cfg.train.max_seq_len,
        tokenizer_path=tokenizer_path,
        limit=cfg.dataset.limit,
    )

    val_dataset = MIDIDatasetPresaved(
        split="validation",
        max_seq_len=cfg.train.max_seq_len,
        tokenizer_path=tokenizer_path,
        limit=cfg.dataset.limit,
    )

    # Initialize model
    model = MIDITransformerEncoder(
        vocab_size=viz_dataset.vocab_size,
        embed_dim=cfg.train.embed_dim,
        nhead=cfg.train.nhead,
        num_layers=cfg.train.num_layers,
        max_seq_len=cfg.train.max_seq_len,
        dropout=cfg.train.dropout,
    ).to(device)

    # Train the model
    model = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=cfg.train,
        device=device,
        model_name=cfg.train.model_name,
        wandb_project=cfg.wandb.project,
        viz_dataset=viz_dataset,
        viz_interval=cfg.wandb.viz_interval,
    )

    # Load best checkpoint
    best_checkpoint = torch.load(f"models/{cfg.train.model_name}.pth", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()

    # Evaluate the model on the test dataset
    test_dataset = MIDIDatasetDynamic(
        split="test",
        max_seq_len=cfg.train.max_seq_len,
        tokenizer_path=tokenizer_path,
    )

    test_loss, perplexity = evaluate(model, test_dataset, device)
    print(f"Test Loss: {test_loss: .4f}, Perplexity: {perplexity: .4f}")

    # Visualize embeddings
    print("Visualizing embeddings...")
    visualize_embeddings(
        model=model,
        device=device,
        max_seq_len=cfg.train.max_seq_len,
        file_name="final_embeddings.html",
        dataset=viz_dataset,
    )
