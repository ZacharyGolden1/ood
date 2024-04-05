"""
Helper functions for training and testing the model
"""

from typing import Tuple, Dict, List
import torch
from tqdm import tqdm


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    One epoch of training

    TODO: accuracy metric

    Returns: (train_loss, train_accuracy)
    """
    model.train()

    train_loss, train_acc = 0.0, 0.0

    for batch, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(dataloader), train_acc / len(dataloader)


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    One epoch of testing

    TODO: accuracy metric

    Returns: (test_loss, test_accuracy)
    """
    model.eval()

    test_loss, test_acc = 0.0, 0.0

    with torch.no_grad():
        for batch, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            test_loss += loss.item()

    return test_loss / len(dataloader), test_acc / len(dataloader)


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
) -> Dict[str, List[float]]:
    """
    Train and test the model

    Returns:
       {train_loss: [...],
        train_acc: [...],
        test_loss: [...],
        test_acc: [...]}
    """
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model, train_dataloader, loss_fn, optimizer, device
        )
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)

        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        # TODO: log to wandb

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}"
        )

    return results