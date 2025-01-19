import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR

from midi_embeddings.dataset import MIDIDataset
from midi_embeddings.transformer import MIDITransformerEncoder, generate_causal_mask


def train_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    scaler,
    scheduler,
    epoch,
    num_epochs,
):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

    for batch_idx, batch in enumerate(progress_bar):
        batch = batch.to(device)  # [batch_size, seq_len]

        # Prepare input and target sequences
        input_seq = batch[:, :-1]  # Input sequence
        target_seq = batch[:, 1:]  # Target sequence

        # Generate causal mask
        src_mask = generate_causal_mask(input_seq.size(1), device)

        # Forward pass
        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            outputs = model(input_seq, src_mask)
            # Compute loss
            loss = criterion(outputs.view(-1, model.vocab_size), target_seq.reshape(-1))

        # Backpropagation
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Update total loss
        total_loss += loss.item()
        current_lr = scheduler.get_last_lr()[0]

        progress_bar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / (batch_idx + 1):.4f}",
                "lr": f"{current_lr:.2e}",
            }
        )

    return total_loss / len(dataloader)


def validate(
    model,
    dataloader,
    criterion,
    device,
):
    model.eval()
    total_loss = 0.0
    perplexity = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            batch = batch.to(device)
            input_seq = batch[:, :-1]
            target_seq = batch[:, 1:]

            src_mask = generate_causal_mask(input_seq.size(1), device)

            outputs = model(input_seq, src_mask)
            loss = criterion(outputs.view(-1, model.vocab_size), target_seq.reshape(-1))

            total_loss += loss.item()
            perplexity += torch.exp(loss).item()

    return total_loss / len(dataloader), perplexity / len(dataloader)


def train_model(
    model,
    train_dataset,
    val_dataset,
    config,
    device,
    model_name="model",
):
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
    )

    # Initialize training components
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.95),
    )

    # Calculate total steps for scheduler
    total_steps = len(train_loader) * config["epochs"]

    scheduler = OneCycleLR(
        optimizer,
        max_lr=config["learning_rate"],
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler()

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(config["epochs"]):
        # Train
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler,
            epoch=epoch,
            num_epochs=config["epochs"],
        )

        # Validate
        val_loss, perplexity = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{config['epochs']} - "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Perplexity: {perplexity:.4f} "
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_loss,
                    "config": config,
                },
                f"models/{model_name}.pth",
            )
            print("New best model saved!")

    return model


def main():
    # Configuration
    config = {
        "max_seq_len": 2048,
        "embed_dim": 384,
        "nhead": 6,
        "num_layers": 4,
        "batch_size": 8,
        "epochs": 100,
        "learning_rate": 1e-4,
        "weight_decay": 1e-3,
        "dropout": 0.3,
    }

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42)

    total_samples = 962
    indices = torch.randperm(total_samples).tolist()
    val_size = int(0.1 * total_samples)
    train_size = total_samples - val_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Initialize dataset
    train_dataset = MIDIDataset(
        tokens_path="data/tokens.json",
        indices=train_indices,
        max_seq_len=config["max_seq_len"],
    )

    val_dataset = MIDIDataset(
        tokens_path="data/tokens.json",
        indices=val_indices,
        max_seq_len=config["max_seq_len"],
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

    # Train model
    model = train_model(
        model,
        train_dataset,
        val_dataset,
        config,
        device,
        model_name="model_2048",
    )


if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
    torch.cuda.empty_cache()
