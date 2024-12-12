import torch
import torch.nn as nn
from torch.utils.data import Subset
from modules import HeteroGNN
from dataset import GraphDataset
from sklearn.model_selection import train_test_split
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
)
from pandas import DataFrame

import argparse
from tqdm import tqdm
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", "-e", metavar="E", default=10, type=int, help="Number of epochs"
    )
    parser.add_argument(
        "--learning-rate",
        "-lr",
        metavar="LR",
        default=1e-3,
        type=float,
        help="Learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--grad-clip",
        "-gc",
        metavar="GC",
        default=1.0,
        type=float,
        help="Gradient clipping",
        dest="grad_clip",
    )
    parser.add_argument(
        "--val-percent",
        "-val",
        metavar="VAL",
        default=0.1,
        type=float,
        help="Validation sampled from training set",
        dest="val_percent",
    )
    parser.add_argument(
        "--weight-decay",
        "-wd",
        metavar="WD",
        default=0,
        type=float,
        help="Weight decay",
        dest="wd",
    )
    parser.add_argument(
        "--grad-accum",
        "-ga",
        metavar="GA",
        default=1,
        type=int,
        help="Number of gradient accumulation over batches",
        dest="grad_accum",
    )
    return parser.parse_args()


@torch.inference_mode()
def evaluate(model: nn.Module, dataset: GraphDataset, device: torch.device):
    acc = MulticlassAccuracy(num_classes=3, average="none").to(device)
    f1 = MulticlassF1Score(num_classes=3, average="none").to(device)
    conf_mat = MulticlassConfusionMatrix(num_classes=3).to(device)

    model.eval().to(device)
    for i, sample in tqdm(
        enumerate(dataset), desc="Evaluation round", total=len(dataset), leave=True
    ):
        input = sample["data"].to(device)
        label = sample["label"].unsqueeze(0).to(device)

        output = model(input.x_dict, input.edge_index_dict)
        acc.update(output, label)
        f1.update(output, label)
        conf_mat.update(output, label)

    model.train()
    scores = (
        DataFrame(
            {
                "Accuracy": acc.compute().cpu().numpy(),
                "F1 Score": f1.compute().cpu().numpy(),
            }
        ).T
        * 100
    )
    scores.rename(columns={0: "bad", 1: "normal", 2: "good"}, inplace=True)
    scores["Macro-avg"] = scores.mean(axis=1)

    conf_mat = DataFrame(
        conf_mat.compute().cpu().numpy(),
        columns=["Pred bad", "Pred normal", "Pred good"],
        index=["True bad", "True normal", "True good"],
    )

    return {
        "Scores": scores,
        "Confusion Matrix": conf_mat,
    }


if __name__ == "__main__":
    args = get_args()

    dataset = GraphDataset(
        "/home/ptdat/Desktop/graph/data/processed", 
        label_csv="/home/ptdat/Desktop/graph/data/course_labeled.csv")
    labels = [sample["label"] for sample in dataset]
    n_samples = len(dataset)

    train_idx, val_idx = train_test_split(
        range(n_samples), test_size=args.val_percent, stratify=labels
    )
    train_set = Subset(dataset, indices=train_idx)
    val_set = Subset(dataset, indices=val_idx)
    n_train = len(train_set)
    n_val = len(val_set)

    model = HeteroGNN(hidden_channels=128, out_channels=3, num_layers=6)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        with tqdm(
            total=n_train,
            desc=f"Epoch {epoch}/{args.epochs}",
            unit="course",
        ) as pbar:
            for i, sample in enumerate(train_set):
                input = sample["data"].to(device)
                label = sample["label"].unsqueeze(0).to(device)

                output = model(input.x_dict, input.edge_index_dict)
                loss = loss_fn(output, label)
                loss.backward()

                if i % args.grad_accum == 0:
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad /= args.grad_accum

                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                pbar.update(1)
                pbar.set_postfix_str(f"loss = {loss.item():.4f}")

            evals = evaluate(
                model=model,
                dataset=val_set,
                device=device,
            )
            for name, eval in evals.items():
                print(f"\n{name}")
                print(eval)

        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/epoch-{epoch}.pth")
        print(f"Checkpoint {epoch} saved")
