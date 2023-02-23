from sklearn.preprocessing import LabelEncoder
from surprise import Dataset as SupriseDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


def load_data(config):
    test_size = config["test_size"]
    data = SupriseDataset.load_builtin("ml-100k")

    dtypes = {"user_id": int, "item_id": int, "rating": float}

    data = pd.DataFrame(data.raw_ratings, columns=["user_id", "item_id", "rating", "timestamp"])
    data = data.drop(["timestamp"], axis=1)
    data = data.astype(dtypes)

    encoder = LabelEncoder()
    data["user_id"] = encoder.fit_transform(data["user_id"])
    data["item_id"] = encoder.fit_transform(data["item_id"])

    n_user = data["user_id"].nunique()
    n_item = data["item_id"].nunique()

    train_data, test_data = train_test_split(data, test_size=test_size)

    return train_data, test_data, n_user, n_item


class NCFDataset(Dataset):
    def __init__(self, interactions):
        super().__init__()
        self.interactions = interactions

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        interaction = self.interactions.iloc[idx]

        user_id = interaction["user_id"]
        item_id = interaction["item_id"]
        rating = interaction["rating"]

        return user_id.astype(int), item_id.astype(int), rating
