import torch

from midi_embeddings.train import evaluate, train_model
from midi_embeddings.visualize import visualize_embeddings
from midi_embeddings.transformer import MIDITransformerEncoder
from midi_embeddings.dataset import MIDIDatasetDynamic, MIDIDatasetPresaved


def main():
    # Configuration settings
    config = {
        "max_seq_len": 2048,
        "embed_dim": 348,
        "nhead": 6,
        "num_layers": 4,
        "batch_size": 8,
        "epochs": 100,
        "learning_rate": 1e-4,
        "weight_decay": 1e-3,
        "dropout": 0.3,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # Initialize datasets
    train_dataset = MIDIDatasetPresaved(
        split="train",
        max_seq_len=config["max_seq_len"],
        tokenizer_path="awesome.json",
    )

    val_dataset = MIDIDatasetPresaved(
        split="validation",
        max_seq_len=config["max_seq_len"],
        tokenizer_path="awesome.json",
    )

    test_dataset = MIDIDatasetDynamic(
        split="test",
        max_seq_len=config["max_seq_len"],
        tokenizer_path="awesome.json",
    )

    # Initialize model
    model = MIDITransformerEncoder(
        vocab_size=train_dataset.vocab_size,
        embed_dim=config["embed_dim"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        max_seq_len=config["max_seq_len"],
        dropout=config["dropout"],
    ).to(device)

    # Train the model
    model = train_model(
        model,
        train_dataset,
        val_dataset,
        config,
        device,
        model_name="model_new.pth",
    )

    test_loss, perplexity = evaluate(model, test_dataset, device)
    print(f"Test Loss: {test_loss: .4f}, Perplexity: {perplexity: .4f}")

    # Visualize embeddings
    visualize_embeddings(model, device, max_seq_len=config["max_seq_len"], file_name="embeddings.html", limit=1000)


if __name__ == "__main__":
    main()
    torch.cuda.empty_cache()
