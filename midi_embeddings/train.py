from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader

if torch.cuda.is_available():
    from torch.amp import GradScaler

from midi_embeddings.transformer import generate_causal_mask


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Union[nn.Module, callable],
    optimizer: optim.Optimizer,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    num_epochs: int,
) -> float:
    """Trains the model for one epoch

    Args:
        model (nn.Module): Model for training
        dataloader (Dataloader): DataLoader for training
        criterion (nn.Module / callable): loss function
        optimizer (optim.Optimizer): optimizer to be used during training
        device (torch.device): device to train the model on
        scheduler (optim.lr_scheduler): lr schedule
        epoch (int): current epoch
        num_epochs (int): total number of epochs

    Returns:
        float: Average loss for the epoch
    """
    model.train()
    total_loss = 0.0

    if torch.cuda.is_available():
        scaler = GradScaler()
    else:
        scaler = None

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

    for batch_idx, batch in enumerate(progress_bar):
        batch = batch.to(device)  # [batch_size, seq_len]

        # Prepare input and target sequences
        input_seq = batch[:, :-1]
        target_seq = batch[:, 1:]

        src_mask = generate_causal_mask(input_seq.size(1), device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                # Forward pass
                outputs = model(input_seq, src_mask)
                # Compute loss
                loss = criterion(outputs.view(-1, model.vocab_size), target_seq.reshape(-1))

            # Backprop
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

        else:
            # Forward pass
            outputs = model(input_seq, src_mask)
            # Compute loss
            loss = criterion(outputs.view(-1, model.vocab_size), target_seq.reshape(-1))

            # Backprop
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        # Update total loss
        total_loss += loss.item()
        current_lr = scheduler.get_last_lr()[0]

        progress_bar.set_postfix(
            {
                "loss": f"{loss.item(): .4f}",
                "avg_loss": f"{total_loss / (batch_idx + 1): .4f}",
                "lr": f"{current_lr: .2e}",
            }
        )

    return total_loss / len(dataloader)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Union[nn.Module, callable],
    device: torch.device,
) -> Union[float, float]:
    """Evaluates current model performance during training

    Args:
        model (nn.Module): model for validation
        dataloader (DataLoader): dataloader for validation
        criterion (Union[nn.Module, callable]): loss function to be used during validation
        device (torch.device): device to validate the model on

    Returns:
        Union[float, float]: loss, perplexity
    """
    model.eval()
    total_loss = 0.0
    perplexity = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            batch = batch.to(device)

            # Prepare input and target sequences
            input_seq = batch[:, :-1]
            target_seq = batch[:, 1:]

            src_mask = generate_causal_mask(input_seq.size(1), device)

            # Pass through model and calculate: loss, perplexity
            outputs = model(input_seq, src_mask)
            loss = criterion(outputs.view(-1, model.vocab_size), target_seq.reshape(-1))

            total_loss += loss.item()
            perplexity += torch.exp(loss).item()

    return total_loss / len(dataloader), perplexity / len(dataloader)


def train_model(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: dict,
    device: torch.device,
    model_name: str = "model",
    optimizer: optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    start_epoch: int = 0,
) -> nn.Module:
    """Trains the model using defined configuration

    Args:
        model (nn.Module): Model for training
        train_dataset (torch.utils.data.Dataset): Training dataset
        val_dataset (torch.utils.data.Dataset): Validation dataset
        config (dict): Configuration file
        device (torch.device): Device to train on
        model_name (str, optional): File name for the model to be saved. Defaults to "model".
        optimizer (optim.Optimizer, optional): Existing optimizer. Defaults to None.
        scheduler (torch.optim.lr_scheduler, optional): Existing scheduler. Defaults to None.
        start_epoch (int, optional): Starting epoch for resuming training. Defaults to 0.

    Returns:
        nn.Module: Trained model.
    """
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

    # Initialize optimizer
    if optimizer is None:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            betas=(0.9, 0.95),
        )

    # Initialize scheduler
    if scheduler is None:
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

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(start_epoch, start_epoch + config["epochs"]):
        # Training phase
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            num_epochs=start_epoch + config["epochs"],
        )

        # Validation phase
        val_loss, perplexity = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{start_epoch + config['epochs']} - "
            f"Train Loss: {train_loss: .4f}, "
            f"Val Loss: {val_loss: .4f}, "
            f"Perplexity: {perplexity: .4f} "
        )

        Path("models").mkdir(exist_ok=True)

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


def evaluate(model: nn.Module, test_dataset: torch.utils.data.Dataset, device: torch.device) -> float:
    """Evaluates the model on the test dataset

    Args:
        model (nn.Module): Model to evaluate
        test_dataset (torch.utils.data.Dataset): Test dataset
        device (torch.device): Device to evaluate the model on

    Returns:
        tuple: test_loss, perplexity
    """
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
    )

    criterion = nn.CrossEntropyLoss()
    test_loss, perplexity = validate(model, test_loader, criterion, device)

    return test_loss, perplexity


def resume_training(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    train_dataset: Dataset,
    val_dataset: Dataset,
) -> nn.Module:
    """Resume training from a checkpoint

    Args:
        model (nn.Module): Model to resume training
        checkpoint_path (str): Path to the checkpoint
        device (torch.device): Device to resume training on
        train_dataset (torch.utils.data.Dataset): Training dataset
        val_dataset (torch.utils.data.Dataset): Validation dataset

    Returns:
        nn.Module: Resumed model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    config = checkpoint["config"]
    start_epoch = checkpoint["epoch"] + 1  # Start from the next epoch

    # Initialize optimizer and load state
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.95),
    )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Initialize scheduler and load state
    batches_count = len(train_dataset) // config["batch_size"]
    total_steps = batches_count * config["epochs"]
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config["learning_rate"],
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
    )
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Resume training
    model = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        device=device,
        model_name=f"{Path(checkpoint_path).stem}_resumed",
        optimizer=optimizer,
        scheduler=scheduler,
        start_epoch=start_epoch,
    )

    return model
