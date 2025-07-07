import argparse
import logging
import os
from typing import List, Tuple

import torch
import torch.distributed as dist
import wandb
from datasets import Dataset, load_dataset
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import models
import utils
from audio_data import UrbanSoundDataset

# --- Hyperparameters ---
hyperparameters = {
    "batch_size": 512,
    "learning_rate": 8e-4,
    "epochs": 30,
    "patience": 6,
    "seed": 42,
    "dropout": 0.15,
    "weight_decay": 1e-4,
}

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Train a CNN for audio classification.")
parser.add_argument("--entity", default="mlx-institute", help="WandB entity")
parser.add_argument("--project", default="cnn-classifier", help="WandB project")
parser.add_argument("--check", action="store_true", help="Run a quick check with one fold.")


# --- DDP Helper Functions ---

def setup_ddp():
    """Initializes the distributed process group."""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup_ddp():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()


def is_main_process():
    """Checks if the current process is the main one (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0


# --- Core Functions ---

def get_model_file_path(test_fold: int) -> str:
    """Generates a unique path for each fold's checkpoint."""
    return models.CNN_MODEL_PATH.replace(".pth", f"_fold_{test_fold + 1}.pth")


def setup_data() -> Tuple[Dataset, List[List[int]]]:
    """Loads the dataset and prepares fold indices."""
    raw_data = load_dataset("danavery/urbansound8K")
    hf_dataset = raw_data['train']
    fold_indices = [[] for _ in range(10)]
    for idx, sample in enumerate(hf_dataset):
        fold_indices[sample['fold'] - 1].append(idx)
    return UrbanSoundDataset(hf_dataset), fold_indices


def get_fold_dataloaders(
        dataset: Dataset, fold_indices: List[List[int]], test_fold: int, val_fold: int, config: dict
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Creates DataLoaders for a specific fold."""
    test_indices = fold_indices[test_fold]
    val_indices = fold_indices[val_fold]
    train_indices = [idx for i in range(10) if i not in [test_fold, val_fold] for idx in fold_indices[i]]

    train_subset, val_subset, test_subset = Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(
        dataset, test_indices)

    ## DDP Note: Use DistributedSampler if DDP is active.
    is_ddp = dist.is_initialized()
    train_sampler = DistributedSampler(train_subset) if is_ddp else None
    val_sampler = DistributedSampler(val_subset, shuffle=False) if is_ddp else None

    device = utils.get_device()
    pin_memory = device.type == "cuda"
    num_workers = min(os.cpu_count(), 8) if pin_memory else 0

    train_dl = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=(train_sampler is None),
                          sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
    val_dl = DataLoader(val_subset, batch_size=config["batch_size"], shuffle=False, sampler=val_sampler,
                        num_workers=num_workers, pin_memory=pin_memory)
    test_dl = DataLoader(test_subset, batch_size=config["batch_size"], shuffle=False, num_workers=num_workers,
                         pin_memory=pin_memory)

    return train_dl, val_dl, test_dl


def run_batch(model, dataloader, loss_fn, device, optimizer=None, desc=""):
    """Runs one pass over the dataloader."""
    is_train = optimizer is not None
    model.train(is_train)

    ## DDP Note: Ensure different shuffling for each epoch in DDP.
    if isinstance(dataloader.sampler, DistributedSampler):
        dataloader.sampler.set_epoch(int(desc.split(" ")[-1]) - 1)

    total_loss, correct, total_samples = 0.0, 0, 0

    iterator = tqdm(dataloader, desc=desc, disable=not is_main_process(), leave=False)
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for X, y in iterator:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * X.size(0)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            total_samples += X.size(0)

    avg_loss = total_loss / total_samples
    accuracy = 100 * correct / total_samples
    return accuracy, avg_loss


def run_fold_training(model, train_dl, val_dl, loss_fn, optimizer, config, device, test_fold):
    """Manages the training loop with early stopping for a single fold."""
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(config["epochs"]):
        train_acc, train_loss = run_batch(model, train_dl, loss_fn, device, optimizer, f"Training epoch {epoch + 1}")
        val_acc, val_loss = run_batch(model, val_dl, loss_fn, device, desc=f"Validating epoch {epoch + 1}")

        if is_main_process():
            wandb.log({
                f"fold_{test_fold + 1}/train_loss": train_loss, f"fold_{test_fold + 1}/train_acc": train_acc,
                f"fold_{test_fold + 1}/val_loss": val_loss, f"fold_{test_fold + 1}/val_acc": val_acc,
            })
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                ## DDP Note: Unwrap the model before saving the state_dict.
                model_to_save = model.module if isinstance(model, DDP) else model
                torch.save(model_to_save.state_dict(), get_model_file_path(test_fold))
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= config["patience"]:
                logging.info(f"Early stopping triggered at epoch {epoch + 1}.")
                break
    return model


def run_single_fold(dataset, fold_indices, test_fold, config, device):
    """Runs training and testing for a single cross-validation fold."""
    val_fold = (test_fold + 1) % 10
    train_dl, val_dl, test_dl = get_fold_dataloaders(dataset, fold_indices, test_fold, val_fold, config)

    if is_main_process():
        logging.info(
            f"Fold {test_fold + 1}: Train={len(train_dl.dataset)}, Val={len(val_dl.dataset)}, Test={len(test_dl.dataset)}")

    model = models.CNN().to(device)
    ## DDP Note: Wrap model in DDP if active. `find_unused_parameters` can help with complex models.
    if dist.is_initialized():
        model = DDP(model, device_ids=[device], find_unused_parameters=False)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    model = run_fold_training(model, train_dl, val_dl, loss_fn, optimizer, config, device, test_fold)

    # Load the best performing model for testing
    if is_main_process():
        model_to_test = model.module if isinstance(model, DDP) else model
        model_to_test.load_state_dict(torch.load(get_model_file_path(test_fold), map_location=device))
        test_acc, _ = run_batch(model_to_test, test_dl, loss_fn, device, desc=f"Testing fold {test_fold + 1}")
        logging.info(f"Fold {test_fold + 1} test accuracy: {test_acc:.2f}%")
        return test_acc
    return None


def main():
    """Main function to orchestrate the training and validation process."""
    args = parser.parse_args()
    utils.setup_logging()

    ## DDP Note: Check for environment variables set by torchrun to setup DDP.
    is_ddp = "WORLD_SIZE" in os.environ
    if is_ddp:
        setup_ddp()

    device = torch.device(
        f"cuda:{os.environ['LOCAL_RANK']}" if is_ddp else "cuda" if torch.cuda.is_available() else "cpu")

    if is_main_process():
        wandb.init(entity=args.entity, project=args.project, config=hyperparameters)

    dataset, fold_indices = setup_data()
    fold_accuracies = []
    num_folds = 1 if args.check else 10

    for test_fold in range(num_folds):
        if is_main_process():
            logging.info(f"\n{'=' * 50}\nStarting Fold {test_fold + 1}/{num_folds}\n{'=' * 50}")

        fold_acc = run_single_fold(dataset, fold_indices, test_fold, hyperparameters, device)
        if is_main_process() and fold_acc is not None:
            fold_accuracies.append(fold_acc)

    if is_main_process():
        mean_acc = sum(fold_accuracies) / len(fold_accuracies)
        std_acc = (sum((x - mean_acc) ** 2 for x in fold_accuracies) / len(fold_accuracies)) ** 0.5
        logging.info(f"\n{'=' * 50}\n10-Fold CV Results: {mean_acc:.2f}% ± {std_acc:.2f}%\n{'=' * 50}")
        wandb.log({"mean_test_accuracy": mean_acc, "std_test_accuracy": std_acc})
        wandb.finish()

    if is_ddp:
        cleanup_ddp()

if __name__ == "__main__":
    main()
