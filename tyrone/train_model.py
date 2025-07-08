import argparse
import logging
import os
from typing import List, Tuple

import json
import torch
from torch.utils.data import Dataset

import wandb
from torch import nn, optim, distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import models
import utils

# --- Hyperparameters ---
hyperparameters = {
    "batch_size": 512,
    "learning_rate": 8e-4,
    "epochs": 30,
    "patience": 3,
    "seed": 42,
    "dropout": 0.15,
    "weight_decay": 1e-4,
    ## Transformer only
    "model_dim": 256,
    "ffn_dim": 512,
    "num_heads": 8,
    "num_encoders": 4,
    "patch_size": 8,
}

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Train a model for audio classification.")
parser.add_argument("--entity", default="mlx-institute", help="WandB entity")
parser.add_argument(
    "--check", action="store_true", help="Run a quick check with one epoch."
)
parser.add_argument("--encoder", action="store_true", help="Use the encoder or the cnn")
args = parser.parse_args()


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


def get_model_file_path(test_fold: int) -> str:
    """Generates a unique path for each fold's checkpoint."""
    base = models.ENCODER_MODEL_PATH if args.encoder else models.CNN_MODEL_PATH
    return base.replace(".pth", f"_fold_{test_fold + 1}.pth")


def save_ckpt(state_dict, path: str):
    """
    Save `state_dict` safely in a DDP job.
    Only rank‑0 writes, with an atomic rename to avoid partial files.
    """
    # Make sure all ranks reach this point
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    if rank == 0:
        tmp_path = path + ".tmp"
        torch.save(state_dict, tmp_path)
        os.replace(tmp_path, path)


class UrbanSoundDataset(Dataset):
    def __init__(self, metadata_path):
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item_info = self.metadata[idx]
        spectrogram = torch.load(item_info["path"], weights_only=True)
        class_id = item_info["class_id"]
        return spectrogram, class_id


def setup_data():
    """Loads the dataset using the preprocessed metadata."""
    metadata_path = f"{utils.SPECTROGRAM_DIR}/metadata.json"
    full_dataset = UrbanSoundDataset(metadata_path=metadata_path)

    # The fold information is now in the metadata, so we create fold indices from it.
    fold_indices = [[] for _ in range(10)]
    for idx, item in enumerate(full_dataset.metadata):
        # Folds are 1-10 in the data, so convert to 0-9 index.
        fold_num = item["fold"] - 1
        fold_indices[fold_num].append(idx)

    return full_dataset, fold_indices


def get_fold_dataloaders(
    dataset: Dataset,
    fold_indices: List[List[int]],
    test_fold: int,
    val_fold: int,
    config: dict,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Creates DataLoaders for a specific fold."""
    test_indices = fold_indices[test_fold]
    val_indices = fold_indices[val_fold]
    train_indices = [
        idx
        for i in range(10)
        if i not in [test_fold, val_fold]
        for idx in fold_indices[i]
    ]

    train_subset, val_subset, test_subset = (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
    )

    ## DDP Note: Use DistributedSampler if DDP is active.
    is_ddp = dist.is_initialized()
    train_sampler = DistributedSampler(train_subset) if is_ddp else None
    val_sampler = DistributedSampler(val_subset, shuffle=False) if is_ddp else None

    device = utils.get_device()
    pin_memory = device.type == "cuda"
    num_workers = min(os.cpu_count(), 8) if pin_memory else 0

    train_dl = DataLoader(
        train_subset,
        batch_size=config["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_dl = DataLoader(
        val_subset,
        batch_size=config["batch_size"],
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_dl = DataLoader(
        test_subset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

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
    maybe_autocast, scaler = utils.amp_components(device, train=is_train)

    with context:
        for X, y in iterator:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with maybe_autocast:
                pred = model(X)
                loss = loss_fn(pred, y)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * X.size(0)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            total_samples += X.size(0)

    avg_loss = total_loss / total_samples
    accuracy = 100 * correct / total_samples
    return accuracy, avg_loss


def run_fold_training(
    model, train_dl, val_dl, loss_fn, optimizer, config, device, test_fold
):
    """Manages the training loop with early stopping for a single fold."""
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(config["epochs"]):
        train_acc, train_loss = run_batch(
            model, train_dl, loss_fn, device, optimizer, f"Training epoch {epoch + 1}"
        )
        val_acc, val_loss = run_batch(
            model, val_dl, loss_fn, device, desc=f"Validating epoch {epoch + 1}"
        )

        if is_main_process():
            wandb.log(
                {
                    f"fold_{test_fold + 1}/train_loss": train_loss,
                    f"fold_{test_fold + 1}/train_acc": train_acc,
                    f"fold_{test_fold + 1}/val_loss": val_loss,
                    f"fold_{test_fold + 1}/val_acc": val_acc,
                }
            )

        if dist.is_initialized():
            t = torch.tensor(val_loss, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            val_loss = (t / dist.get_world_size()).item()
            dist.barrier()
        is_best = val_loss < best_val_loss
        if is_best and is_main_process():
            best_val_loss = val_loss
            epochs_no_improve = 0
            ## DDP Note: Unwrap the model before saving the state_dict.
            model_to_save = model.module if isinstance(model, DDP) else model
            save_ckpt(model_to_save.state_dict(), get_model_file_path(test_fold))
        else:
            epochs_no_improve += 1
        if dist.is_initialized():
            dist.barrier()

        early_stop_local = is_main_process() and (
            epochs_no_improve >= config["patience"] or args.check
        )
        early_stop_tensor = torch.tensor(
            [early_stop_local], device=device, dtype=torch.uint8
        )

        if dist.is_initialized():
            # share the decision (bool -> int tensor) with all ranks
            dist.broadcast(early_stop_tensor, src=0)

        early_stop = bool(early_stop_tensor.item())
        if early_stop:
            break
    return model


def get_model(config):
    if args.encoder:
        return models.AudioTransformer(
            model_dim=config["model_dim"],
            ffn_dim=config["ffn_dim"],
            num_heads=config["num_heads"],
            num_encoders=config["num_encoders"],
            dropout=config["dropout"],
            patch_size=config["patch_size"],
        )
    return models.CNN()


def run_single_fold(dataset, fold_indices, test_fold, config, device):
    """Runs training and testing for a single cross-validation fold."""
    val_fold = (test_fold + 1) % 10
    train_dl, val_dl, test_dl = get_fold_dataloaders(
        dataset, fold_indices, test_fold, val_fold, config
    )

    if is_main_process():
        logging.info(
            f"Fold {test_fold + 1}: Train={len(train_dl.dataset)}, Val={len(val_dl.dataset)}, Test={len(test_dl.dataset)}"
        )

    model = get_model(config).to(device)
    ## DDP Note: Wrap model in DDP if active. `find_unused_parameters` can help with complex models.
    if dist.is_initialized():
        model = DDP(model, device_ids=[device], find_unused_parameters=False)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    model = run_fold_training(
        model, train_dl, val_dl, loss_fn, optimizer, config, device, test_fold
    )

    if dist.is_initialized():
        dist.barrier()
    # Load the best performing model for testing
    if is_main_process():
        model_to_test = model.module if isinstance(model, DDP) else model
        model_to_test.load_state_dict(
            torch.load(
                get_model_file_path(test_fold), map_location=device, weights_only=True
            )
        )
        test_acc, _ = run_batch(
            model_to_test,
            test_dl,
            loss_fn,
            device,
            desc=f"Testing fold {test_fold + 1}",
        )
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
        f"cuda:{os.environ['LOCAL_RANK']}"
        if is_ddp
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    if is_main_process():
        project = "encoder-classifier" if args.encoder else "cnn-classifier"
        wandb.init(entity=args.entity, project=project, config=hyperparameters)

    dataset, fold_indices = setup_data()
    fold_accuracies = []
    num_folds = 1  # to save time only one fold

    for test_fold in range(num_folds):
        if is_main_process():
            logging.info(
                f"\n{'=' * 50}\nStarting Fold {test_fold + 1}/{num_folds}\n{'=' * 50}"
            )

        fold_acc = run_single_fold(
            dataset, fold_indices, test_fold, hyperparameters, device
        )
        if is_main_process() and fold_acc is not None:
            fold_accuracies.append(fold_acc)

    if is_main_process():
        mean_acc = sum(fold_accuracies) / len(fold_accuracies)
        std_acc = (
            sum((x - mean_acc) ** 2 for x in fold_accuracies) / len(fold_accuracies)
        ) ** 0.5
        logging.info(
            f"\n{'=' * 50}\n10-Fold CV Results: {mean_acc:.2f}% ± {std_acc:.2f}%\n{'=' * 50}"
        )
        wandb.log({"mean_test_accuracy": mean_acc, "std_test_accuracy": std_acc})
        wandb.finish(0)

    if is_ddp:
        cleanup_ddp()


if __name__ == "__main__":
    main()
