import argparse
import torch
import time
import os
import numpy as np
import torch.nn as nn

from ncf import NCF
from data import load_data, NCFDataset
from util import set_seed, read_json
from torch.utils.data import DataLoader
from torch.optim import Adam


def main(config):
    device = config["device"]
    train_data, test_data, n_user, n_item = load_data(config)
    train_dataset = NCFDataset(interactions=train_data)
    test_dataset = NCFDataset(interactions=test_data)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    model_config = config["arch"]
    model = NCF(
        n_user=n_user,
        n_item=n_item,
        embedding_size=model_config["embedding_size"],
        mlp_layer_dims=model_config["mlp_layer_dims"],
        dropout_rate=model_config["dropout_rate"],
        use_gmf=model_config["use_gmf"]
    )
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    model = model.to(config["device"])

    best_val_loss = np.inf
    for epoch in range(config["epochs"]):
        start_time = time.time()

        total_loss = list()
        model.train()
        for user_id, pos_item_id, target in train_loader:
            user_id = user_id.to(device)
            pos_item_id = pos_item_id.to(device)
            target = target.to(device)

            scores = model(user_id, pos_item_id)

            loss = criterion(scores.squeeze().float(), target.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.append(loss.detach().cpu().numpy())

        train_loss = np.mean(total_loss)

        model.eval()
        val_loss = eval(model, criterion, test_loader, device)

        print(f"epoch {epoch} | train_loss {train_loss:.4f} | val_loss {val_loss:.3f} | elapsed_time {time.time() - start_time}")

        if val_loss > best_val_loss:
            best_val_loss = val_loss
            if not os.path.exists(config["save_dir"]):
                os.makedirs(config["save_dir"], exist_ok=True)
            torch.save(model, f"{config['save_dir']}/bpr.pt")


def eval(model, criterion, test_loader, device):
    val_loss = list()
    for user_id, item_id, target in test_loader:
        user_id = user_id.to(device)
        item_id = item_id.to(device)
        target = target.to(device)

        scores = model(user_id, item_id)

        loss = criterion(scores.squeeze(), target)

        val_loss.append(loss.detach().cpu().numpy())

    return np.mean(val_loss)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="BPR")
    args.add_argument(
        "-c",
        "--config",
        default="./config.json",
        type=str,
        help='config file path (default: "./config.json")',
    )
    args = args.parse_args()
    config = read_json(args.config)
    set_seed(seed=config["seed"])

    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    main(config)