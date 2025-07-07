import argparse
import logging
import subprocess
from typing import Optional, Tuple, List

import torch
from datasets import load_dataset, Dataset
from torch import optim, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import models
import utils
import wandb
from audio_data import UrbanSoundDataset

hyperparameters = {
    "batch_size": 384,
    "learning_rate": 5e-4,
    "epochs": 25,
    "patience": 3,
    "model_dim": 256,
    "ffn_dim": 2048,
    "num_encoders": 5,
    "num_heads": 64,
    "seed": 42,
    "dropout": 0.15,
    "weight_decay": 1e-4,
}

sweep_config = {
    "method": "grid",  # can be 'grid', 'random', or 'bayes'
    "metric": {"name": "test_accuracy", "goal": "maximize"},
    "parameters": {
        "batch_size": {"values": [1024]},
        "learning_rate": {"values": [5e-4]},
        "epochs": {"values": [95, 196]},
        "patience": {"values": [-1]},
        "model_dim": {"values": [256]},
        "ffn_dim": {"values": [2048]},
        "num_encoders": {"values": [5]},
        "num_heads": {"values": [64]},
        "dropout": {"values": [0.15]},
        "weight_decay": {"values": [1e-4]},
    },
}

parser = argparse.ArgumentParser(description="Train simple model")
parser.add_argument("--entity", help="W and B entity", default="mlx-institute")
parser.add_argument("--project", help="W and B project", default="cnn-classifier")
parser.add_argument("--sweep", help="Run hyperparameter sweep", action="store_true")
parser.add_argument("--check", help="Make sure it works", action="store_true")
args = parser.parse_args()


def get_git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def setup_data() -> Tuple[Dataset, List[List[int]]]:
    """Setup and return the full dataset and fold indices."""
    raw_data = load_dataset("danavery/urbansound8K")
    hf_dataset = raw_data['train']  # Assuming all data is in 'train' split

    # Group indices by fold
    fold_indices = [[] for _ in range(10)]
    for idx, sample in enumerate(hf_dataset):
        fold_num = sample['fold'] - 1  # Convert 1-10 to 0-9
        fold_indices[fold_num].append(idx)

    full_dataset = UrbanSoundDataset(hf_dataset)

    return full_dataset, fold_indices


def get_fold_dataloaders(
        dataset: Dataset,
        fold_indices: List[List[int]],
        test_fold: int,
        val_fold: int,
        config: dict
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders for a specific fold configuration."""

    # Test fold
    test_indices = fold_indices[test_fold]

    # Validation fold
    val_indices = fold_indices[val_fold]

    # Training folds (all others)
    train_indices = []
    for i in range(10):
        if i != test_fold and i != val_fold:
            train_indices.extend(fold_indices[i])

    # Create subsets
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)

    # Create dataloaders
    device = utils.get_device()
    pin_memory = device.type == "cuda"
    num_workers = 0  # Disable multiprocessing for debugging

    train_dl = DataLoader(
        train_subset,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    val_dl = DataLoader(
        val_subset,
        batch_size=config["batch_size"],
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    test_dl = DataLoader(
        test_subset,
        batch_size=config["batch_size"],
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )

    return train_dl, val_dl, test_dl


def run_batch(
        dataloader,
        model,
        device,
        loss_fn: nn.Module,
        train: bool = False,
        optimizer: Optional[Optimizer] = None,
        desc: str = "",
):
    """
    Runs one pass over `dataloader`.

    If `train` is True, the model is set to training mode and the optimizer is
    stepped. Otherwise the model is evaluated with torch.no_grad().
    Returns (accuracy %, average_loss) for the epoch.
    """
    model.train() if train else model.eval()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    total_loss, correct = 0.0, 0

    iterator = tqdm(dataloader, desc=desc)
    context = torch.enable_grad() if train else torch.no_grad()
    maybe_autocast, scaler = utils.amp_components(device, train)
    with context:
        for X, y in iterator:
            X, y = X.to(device), y.to(device)
            with maybe_autocast:
                pred = model(X)
                loss = loss_fn(pred, y)

            total_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            if train:
                if optimizer is None:
                    raise ValueError("Optimizer must be provided when train=True")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

    avg_loss = total_loss / num_batches
    accuracy = 100 * correct / size
    return accuracy, avg_loss


def run_training(
        model: nn.Module,
        train_dl: DataLoader,
        val_dl: DataLoader,
        device: torch.device,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        config: dict,
) -> nn.Module:
    best_loss = float("inf")
    epochs_since_best = 0
    for epoch in range(config["epochs"]):
        train_correct, train_loss = run_batch(
            dataloader=train_dl,
            model=model,
            device=device,
            train=True,
            loss_fn=loss_fn,
            optimizer=optimizer,
            desc=f"Training epoch {epoch + 1}",
        )
        val_correct, val_loss = run_batch(
            dataloader=val_dl,
            model=model,
            device=device,
            train=False,
            loss_fn=loss_fn,
            desc=f"Validating epoch {epoch + 1}",
        )
        wandb.log(
            {
                "train_accuracy": train_correct,
                "train_loss": train_loss,
                "val_accuracy": val_correct,
                "val_loss": val_loss,
            },
        )
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_since_best = 0
            if not args.check:
                model_dict = {
                    "config": dict(config),  # Ensure config is a plain dict
                    "model_state_dict": model.state_dict(),
                    "best_loss": best_loss,
                }
                torch.save(model_dict, models.CNN_MODEL_PATH)
        else:
            epochs_since_best += 1
        if args.check:
            break
        if config["patience"] == -1:
            continue  # add option to disable early stop
        elif epochs_since_best >= config["patience"]:
            break

    return model


def run_single_fold(
        dataset: Dataset,
        fold_indices: List[List[int]],
        test_fold: int,
        config: dict,
        device: torch.device,
) -> float:
    """Run training and testing for a single fold. Returns test accuracy."""

    # Use fold 9 as validation if test_fold != 9, otherwise use fold 0
    val_fold = 9 if test_fold != 9 else 0

    # Get dataloaders for this fold
    train_dl, val_dl, test_dl = get_fold_dataloaders(
        dataset, fold_indices, test_fold, val_fold, config
    )

    logging.info(f"Fold {test_fold + 1}: Train size={len(train_dl.dataset)}, "
                 f"Val size={len(val_dl.dataset)}, Test size={len(test_dl.dataset)}")

    # Create fresh model for this fold
    model = models.CNN()
    model.to(device)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    # Train the model
    model = run_training(
        model=model,
        train_dl=train_dl,
        val_dl=val_dl,
        device=device,
        loss_fn=loss_fn,
        optimizer=optimizer,
        config=config,
    )

    # Load best checkpoint if available
    if not args.check:
        try:
            fold_model_path = f"{models.CNN_MODEL_PATH}_fold_{test_fold + 1}.pth"
            checkpoint = torch.load(fold_model_path, weights_only=True, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
        except Exception as e:
            logging.warning(f"Failed to load checkpoint for fold {test_fold + 1}: {e}")

    # Test the model
    test_correct, test_loss = run_batch(
        dataloader=test_dl,
        model=model,
        device=device,
        loss_fn=loss_fn,
        train=False,
        desc=f"Testing fold {test_fold + 1}",
    )

    wandb.log({
        f"fold_{test_fold + 1}/test_accuracy": test_correct,
        f"fold_{test_fold + 1}/test_loss": test_loss,
    })

    logging.info(f"Fold {test_fold + 1} test accuracy: {test_correct:.2f}%")
    return test_correct


def run_single_training(config=None):
    """Run a single training session with given config."""
    if config is None:
        config = hyperparameters

    if config["model_dim"] % config["num_heads"] != 0:
        logging.error(
            f"model_dim must be a multiple of num_heads, {config['model_dim']}, {config['num_heads']} not permitted"
        )
        return None
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        config=config,
    )

    device = utils.get_device()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if device.type == "cuda":
        torch.set_float32_matmul_precision("medium")

    # Setup data
    dataset, fold_indices = setup_data()

    fold_accuracies = []
    folds = 1 if args.check else 10
    for test_fold in range(folds):
        logging.info(f"\n{'=' * 50}")
        logging.info(f"Starting fold {test_fold + 1}/{folds}")
        logging.info(f"{'=' * 50}")

        fold_accuracy = run_single_fold(
            dataset=dataset,
            fold_indices=fold_indices,
            test_fold=test_fold,
            config=config,
            device=device,
        )
        fold_accuracies.append(fold_accuracy)

    # Calculate final metrics
    mean_accuracy = sum(fold_accuracies) / len(fold_accuracies)
    std_accuracy = (sum((x - mean_accuracy) ** 2 for x in fold_accuracies) / len(fold_accuracies)) ** 0.5

    # Log final results
    final_results = {
        "mean_test_accuracy": mean_accuracy,
        "std_test_accuracy": std_accuracy,
        "min_test_accuracy": min(fold_accuracies),
        "max_test_accuracy": max(fold_accuracies),
    }

    # Log individual fold results
    for i, acc in enumerate(fold_accuracies):
        final_results[f"fold_{i + 1}_final_accuracy"] = acc

    run.log(final_results)

    logging.info(f"\n{'=' * 60}")
    logging.info(f"10-FOLD CROSS-VALIDATION RESULTS")
    logging.info(f"{'=' * 60}")
    logging.info(f"Mean test accuracy: {mean_accuracy:.2f}% ± {std_accuracy:.2f}%")
    logging.info(f"Min test accuracy: {min(fold_accuracies):.2f}%")
    logging.info(f"Max test accuracy: {max(fold_accuracies):.2f}%")
    logging.info(f"Individual fold accuracies: {[f'{acc:.2f}%' for acc in fold_accuracies]}")

    if not args.check:
        artifact = wandb.Artifact(name="cnn-classifier", type="model")
        artifact.add_file(models.CNN_MODEL_PATH)
        run.log_artifact(artifact)
    run.finish(0, timeout=0)


def main():
    utils.setup_logging()

    if args.sweep:
        sweep_id = wandb.sweep(sweep_config, entity=args.entity, project=args.project)
        wandb.agent(sweep_id, function=run_single_training)
    else:
        config = dict(hyperparameters)  # makes a shallow copy
        config["git_commit"] = get_git_commit()
        run_single_training(config)


if __name__ == "__main__":
    main()
