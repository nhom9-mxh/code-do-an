import torch
from torch.utils.data import Dataset

from pandas import read_csv
from glob import glob
import joblib


class GraphDataset(Dataset):
    LABEL2IDX = {"bad": 0, "normal": 1, "good": 2}

    def __init__(self, data_dir: str, label_csv: str):
        self.samples = glob(f"{data_dir}/*.pkl")
        self.df = read_csv(label_csv)
        self.df["average_completion_rate"].fillna(0.6550275845610755)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = joblib.load(self.samples[index])
        avg_comp = self.df.loc[
            self.df["id"] == sample["_id"], "average_completion_rate"
        ].values[0]
        sample["data"]["course"].x = torch.cat(
            [sample["data"]["course"].x, torch.tensor([[avg_comp]])], dim=1
        ).float()
        sample["label"] = torch.tensor(GraphDataset.LABEL2IDX[sample["label"]])
        return sample
