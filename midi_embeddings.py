import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import os

from midi_embeddings.train import evaluate, train_model
from midi_embeddings.visualize import visualize_embeddings
from midi_embeddings.transformer import MIDITransformerEncoder
from midi_embeddings.dataset import MIDIDatasetDynamic, MIDIDatasetPresaved


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Set device
    DEVICE = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    
    # Set tokenizer path
    tokenizer_path = Path(cfg.dataset.tokenizer_path)
    if not tokenizer_path.exists():
        print(f"Tokenizer at {tokenizer_path} not found. Training required. Please wait.")
        tokenizer_path = None

    # Load dataset for visualization (eval or training)
    viz_dataset = MIDIDatasetPresaved(
        split="all",
        max_seq_len=cfg.train.max_seq_len,
        tokenizer_path=tokenizer_path,
        return_info=True,
        limit=cfg.dataset.limit,
    )
    
    # If tokenizer was null, viz_dataset trained a new one
    tokenizer_path = viz_dataset.tokenizer_path 
    
    if hasattr(cfg, "eval") and cfg.eval.enabled:
        # Load checkpoint
        checkpoint_path = cfg.eval.model_path
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

        print(f"Loading model from {checkpoint_path} for evaluation...")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        
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
            ).to(DEVICE)
        else:
            # Use config from current config file
            model = MIDITransformerEncoder(
                vocab_size=viz_dataset.vocab_size,
                embed_dim=cfg.train.embed_dim,
                nhead=cfg.train.nhead,
                num_layers=cfg.train.num_layers,
                max_seq_len=cfg.train.max_seq_len,
                dropout=cfg.train.dropout,
            ).to(DEVICE)
            
        # Load model weights
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        # Evaluate the model on test dataset
        test_dataset = MIDIDatasetDynamic(
            split="test",
            max_seq_len=cfg.train.max_seq_len,
            tokenizer_path=tokenizer_path,
        )
        
        test_loss, perplexity = evaluate(model, test_dataset, DEVICE)
        print(f"Test Loss: {test_loss:.4f}, Perplexity: {perplexity:.4f}")
        
        # Visualize embeddings if requested
        if cfg.eval.visualize:
            print("Visualizing embeddings...")
            output_file = cfg.eval.output_file or "eval_embeddings.html"
            visualize_embeddings(
                model=model,
                device=DEVICE,
                max_seq_len=cfg.train.max_seq_len,
                file_name=output_file,
                dataset=viz_dataset,
            )
            print(f"Embeddings visualization saved to {output_file}")
        
        return

    # Continue with normal training
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
