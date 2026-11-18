
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.data.features import FEATURE_COLUMNS

torch.manual_seed(42)
np.random.seed(42)

DATA_DIR = Path("data/processed/train")
OUT_DIR = Path("results/ml")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_CANDIDATES = 10
EPOCHS = 15
BATCH_SIZE = 128

class CNN(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

def load_data():
    frames = []
    for p in DATA_DIR.glob("*.parquet"):
        if p.stem.isdigit() and len(p.stem) == 8:
            d = pd.read_parquet(p)
            d = d[d["training_eligible"] == True].copy()
            d["trip_id"] = p.stem
            frames.append(d)

    return pd.concat(frames, ignore_index=True)

def make_sequences(data, scaler=None, fit=False):
    sequences, labels, masks, group_ids = [], [], [], []

    X = data[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)

    if fit:
        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(X)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return X, imputer, scaler

    return X

def main():
    start = time.time()
    print("Loading eligible training rows...", flush=True)
    data = load_data()

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=42
    )
    train_idx, val_idx = next(
        splitter.split(data, groups=data["trip_id"])
    )

    train = data.iloc[train_idx].copy()
    val = data.iloc[val_idx].copy()

    X_train_raw = train[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    X_val_raw = val[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train_raw)
    X_val = imputer.transform(X_val_raw)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    def build_groups(frame, X):
        seqs, ys, masks = [], [], []
        offset = 0

        temp = frame.reset_index(drop=True)
        temp["_row"] = np.arange(len(temp))

        for _, group in temp.groupby(["trip_id", "gps_index"], sort=False):
            group = group.sort_values("candidate_rank")
            indices = group["_row"].to_numpy()

            seq = np.zeros((MAX_CANDIDATES, X.shape[1]), dtype=np.float32)
            y = np.zeros(MAX_CANDIDATES, dtype=np.float32)
            mask = np.zeros(MAX_CANDIDATES, dtype=np.float32)

            count = min(len(group), MAX_CANDIDATES)
            seq[:count] = X[indices[:count]]
            y[:count] = group["label"].astype(int).to_numpy()[:count]
            mask[:count] = 1.0

            seqs.append(seq.T)
            ys.append(y)
            masks.append(mask)

        return (
            torch.tensor(np.array(seqs), dtype=torch.float32),
            torch.tensor(np.array(ys), dtype=torch.float32),
            torch.tensor(np.array(masks), dtype=torch.float32)
        )

    train_x, train_y, train_mask = build_groups(train, X_train)
    val_x, val_y, val_mask = build_groups(val, X_val)

    loader = DataLoader(
        TensorDataset(train_x, train_y, train_mask),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = CNN(len(FEATURE_COLUMNS)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    print(f"Device: {DEVICE}", flush=True)
    print(f"Training GPS observations: {len(train_x):,}", flush=True)
    print(f"Validation GPS observations: {len(val_x):,}", flush=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for xb, yb, mb in loader:
            xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = (criterion(logits, yb) * mb).sum() / mb.sum()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{EPOCHS} "
            f"Loss: {total_loss/len(loader):.4f}",
            flush=True
        )

    model.eval()
    with torch.no_grad():
        logits = model(val_x.to(DEVICE)).cpu().numpy()

    valid = val_mask.numpy().astype(bool)
    y_true = val_y.numpy()[valid]
    probabilities = (1 / (1 + np.exp(-logits)))[valid]
    predictions = (probabilities >= 0.5).astype(int)

    results = {
        "model": "1D CNN",
        "accuracy": accuracy_score(y_true, predictions),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "seconds": round(time.time() - start, 2)
    }

    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_columns": FEATURE_COLUMNS,
        "max_candidates": MAX_CANDIDATES,
        "imputer": imputer,
        "scaler": scaler
    }, OUT_DIR / "cnn_model.pt")

    pd.DataFrame([results]).to_csv(
        OUT_DIR / "cnn_results.csv", index=False
    )

    print("\nCNN RESULTS:", results, flush=True)
    print("Saved CNN model and results.", flush=True)

if __name__ == "__main__":
    main()

