import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from midi_embeddings.train import evaluate, train_model
from midi_embeddings.visualize import visualize_embeddings
from midi_embeddings.transformer import MIDITransformerEncoder
from midi_embeddings.dataset import MIDIDatasetDynamic, MIDIDatasetPresaved


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))  # Print the config for debugging

    # Set device
    DEVICE = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    # Set tokenizer path
    tokenizer_path = Path(cfg.dataset.tokenizer_path)
    if not tokenizer_path.exists():
        print(f"Tokenizer at {tokenizer_path} not found. Training required. Please wait.")
        tokenizer_path = None

    # Load datasets
    train_dataset = MIDIDatasetPresaved(
        split="train",
        max_seq_len=cfg.train.max_seq_len,
        tokenizer_path=tokenizer_path,
        limit=cfg.dataset.limit,
    )
    tokenizer_path = train_dataset.tokenizer_path

    val_dataset = MIDIDatasetPresaved(
        split="validation",
        max_seq_len=cfg.train.max_seq_len,
        tokenizer_path=tokenizer_path,
        limit=cfg.dataset.limit,
    )

    viz_dataset = MIDIDatasetPresaved(
        split="all",
        max_seq_len=cfg.train.max_seq_len,
        tokenizer_path=tokenizer_path,
        return_info=True,
        limit=cfg.dataset.limit,
    )

    # Initialize model
    model = MIDITransformerEncoder(
        vocab_size=train_dataset.vocab_size,
        embed_dim=cfg.train.embed_dim,
        nhead=cfg.train.nhead,
        num_layers=cfg.train.num_layers,
        max_seq_len=cfg.train.max_seq_len,
        dropout=cfg.train.dropout,
    ).to(DEVICE)

    # Train the model
    model = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=cfg.train,
        device=DEVICE,
        model_name=cfg.train.model_name,
        wandb_project=cfg.wandb.project,
        viz_dataset=viz_dataset,
        viz_interval=cfg.wandb.viz_interval,
    )

    # Load best checkpoint
    best_checkpoint = torch.load(f"models/{cfg.train.model_name}.pth", map_location=DEVICE)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()

    # Evaluate the model on the test dataset
    test_dataset = MIDIDatasetDynamic(
        split="test",
        max_seq_len=cfg.train.max_seq_len,
        tokenizer_path=tokenizer_path,
    )

    test_loss, perplexity = evaluate(model, test_dataset, DEVICE)
    print(f"Test Loss: {test_loss:.4f}, Perplexity: {perplexity:.4f}")

    # Visualize embeddings
    print("Visualizing embeddings...")
    visualize_embeddings(
        model=model,
        device=DEVICE,
        max_seq_len=cfg.train.max_seq_len,
        file_name="final_embeddings.html",
        dataset=viz_dataset,
    )


if __name__ == "__main__":
    main()
    torch.cuda.empty_cache()
