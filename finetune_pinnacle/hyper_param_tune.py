# Main finetuning script
import os

print("### check", os.environ["CUDA_VISIBLE_DEVICES"])
import pandas as pd
import numpy as np
import wandb, random

from setup import create_parser, get_hparams, setup_paths
from read_data import load_data
from train_utils import training_and_validation
from metrics_utils import get_metrics, save_torch_train_val_preds, save_results
from data_prep import process_and_split_data

import torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import shuffle

# from sweep_setup import hyper_param

os.environ["OPENBLAS_NUM_THREADS"] = "1"


def create_path(args):
    args.result_dir = os.path.join(args.result_dir, args.task_name)
    args.metrics_output_dir = os.path.join(args.result_dir, args.metrics_output_dir)
    args.models_output_dir = os.path.join(args.result_dir, args.models_output_dir)

    os.makedirs(args.metrics_output_dir, exist_ok=True)
    os.makedirs(args.models_output_dir, exist_ok=True)
    return args


## Setup the args
args = create_parser()
args = create_path(args)

if not args.random:
    np.random.seed(args.random_state)
    random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    torch.cuda.manual_seed(args.random_state)
    torch.backends.cudnn.deterministic = True

##* This is the setup phase
# Set up model environment and data/model paths
models_output_dir, metrics_output_dir, random_state, embed_path, labels_path = (
    setup_paths(args)
)

# Load data
embed, celltype_dict, celltype_protein_dict, positive_proteins, negative_proteins, _ = (
    load_data(
        embed_path,
        labels_path,
        args.positive_proteins_prefix,
        args.negative_proteins_prefix,
        None,
        args.task_name,
    )
)
print("Finished reading data, evaluating...\n")

# Run model
data_split_path = args.data_split_path + ".json"

# Training and validation
X_train, X_test, y_train, y_test, groups_train, cts_train, groups_test = (
    process_and_split_data(
        embed,
        positive_proteins,
        negative_proteins,
        celltype_protein_dict,
        celltype_dict,
        data_split_path,
        random_state=random_state,
        test_size=1 - args.train_size - args.val_size,
    )
)

if not isinstance(X_train, torch.Tensor):
    X_train = torch.from_numpy(X_train)

n_splits = int((args.train_size + args.val_size) / args.val_size)
train_indices, val_indices = list(
    StratifiedGroupKFold(
        n_splits=n_splits, random_state=random_state, shuffle=True
    ).split(X=X_train, groups=groups_train, y=y_train)
)[
    np.random.randint(0, n_splits)
]  # borrow CV generator to generate one split


def main():
    with wandb.init() as run:
        hparams = get_hparams(args)
        config = wandb.config

        hparams = {
            "lr": config.lr,
            "wd": config.wd,
            "hidden_dim_1": config.hidden_dim_1,
            "hidden_dim_2": config.hidden_dim_2,
            "hidden_dim_3": getattr(
                config, "hidden_dim_3", args.hidden_dim_3
            ),  # fallback if not in sweep
            "dropout": config.dropout,
            "actn": config.actn,
            "order": config.order,
            "norm": config.norm,
            "task_name": args.task_name,
        }

        print(hparams)
        print("Batch size", args.batch_size)
        (
            clf,
            best_train_y,
            best_train_preds,
            best_traNO_OF_EPOCHin_cts,
            best_train_groups,
            cts_map_train,
            groups_map_train,
            best_val_y,
            best_val_preds,
            best_val_cts,
            best_val_groups,
            cts_map_val,
            groups_map_val,
            best_epoch,
            best_val_auprc,
        ) = training_and_validation(
            X_train[train_indices],
            X_train[val_indices],
            torch.Tensor(y_train)[train_indices],
            torch.Tensor(y_train)[val_indices],
            np.array(cts_train)[train_indices],
            np.array(cts_train)[val_indices],
            np.array(groups_train)[train_indices],
            np.array(groups_train)[val_indices],
            args.num_epoch,
            args.batch_size,
            args.weigh_sample,
            args.weigh_loss,
            hparams,
            lr_scheduler=args.lr_scheduler,
        )


# 2: Define the search space
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "val AUPRC"},
    "parameters": {
        "norm": {
            "values": [
                "bn",
                "ln",
            ]
        },
        "actn": {"values": ["relu"]},
        "hidden_dim_1": {"values": [8, 16, 32]},
        "hidden_dim_2": {"values": [8, 16, 32]},
        "dropout": {"values": [0.2, 0.5, 0.8]},
        "lr": {"values": [0.01, 0.001, 0.0001]},
        "wd": {"values": [0.001, 0.0001, 0.00001, 0.000001]},
        "order": {"values": ["nd", "dn"]},
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 20,
    },
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="pinnacle_finetune", )

wandb.config.update({"disease_id": args.disease_id}, allow_val_change=True)
# sweep_id = 'mbt7t2yv'


# sweep_id = "yuvaramsingh/pinnacle_finetune/iepr91xt"

wandb.agent(sweep_id, function=main, count=100)
