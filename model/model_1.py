import os
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

# ---------------- Dataset Class ----------------
class TimeSeriesSeq2SeqDataset(Dataset):
    def __init__(self, X, y, decoder_starts):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
        self.decoder_starts = torch.tensor(decoder_starts, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.decoder_starts[idx]

# ---------------- Model Definitions ----------------
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(output_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, input_seq, hidden, cell):
        output, (hidden, cell) = self.lstm(input_seq, (hidden, cell))
        prediction = self.fc(output)
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, decoder_init, tgt_len, y=None):
        hidden, cell = self.encoder(src)
        outputs = []
        decoder_input = decoder_init
        for t in range(tgt_len):
            out, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs.append(out)
            decoder_input = y[:, t:t+1, :]
        return torch.cat(outputs, dim=1)

# ---------------- Data Utilities ----------------
def make_dataset_seq2seq(df, feature_cols, target_col, look_back, tgt_len, ids):
    X, y, first_decoder_inputs = [], [], []
    for pid in ids:
        grp = df[df['profile_id'] == pid].reset_index(drop=True)
        data_f = grp[feature_cols].values
        data_t = grp[target_col].values
        for i in range(len(data_f) - look_back - tgt_len + 1):
            X.append(data_f[i:i + look_back])
            y.append(data_t[i + look_back : i + look_back + tgt_len])
            first_decoder_inputs.append(data_t[i + look_back - 1])
    return np.array(X), np.array(y), np.array(first_decoder_inputs)

def apply_mixed_feature_normalization(df_r, df_s, norm_dict, feature_cols, target_col):
    dfs_combined = []
    profile_ids = sorted(set(df_r['profile_id'].unique()) | set(df_s['profile_id'].unique()))
    for pid in profile_ids:
        row_r = df_r[df_r['profile_id'] == pid].copy().reset_index(drop=True)
        row_s = df_s[df_s['profile_id'] == pid].copy().reset_index(drop=True)
        mixed = pd.DataFrame()
        mixed['profile_id'] = row_r['profile_id']
        for col in feature_cols + [target_col]:
            source = norm_dict[pid][col]
            mixed[col] = row_s[col] if source == "soft" else row_r[col]
        dfs_combined.append(mixed)
    return pd.concat(dfs_combined).reset_index(drop=True)

# ---------------- Training ----------------
def train_model(model, train_loader, val_loader, criterion, optimizer, device, tgt_len, num_epochs=50):
    history = {"train_loss": [], "train_mae": [], "val_loss": [], "val_mae": []}
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_mae = 0, 0
        for x, y, dec in train_loader:
            x, y, dec = x.to(device), y.to(device), dec.to(device)
            optimizer.zero_grad()
            pred = model(x, dec, tgt_len, y)
            loss = criterion(pred, y)
            mae = torch.mean(torch.abs(pred - y))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_mae += mae.item()
        model.eval()
        val_loss, val_mae = 0, 0
        with torch.no_grad():
            for x, y, dec in val_loader:
                x, y, dec = x.to(device), y.to(device), dec.to(device)
                pred = model(x, dec, tgt_len, y)
                loss = criterion(pred, y)
                mae = torch.mean(torch.abs(pred - y))
                val_loss += loss.item()
                val_mae += mae.item()
        n_train, n_val = len(train_loader), len(val_loader)
        history['train_loss'].append(train_loss/n_train)
        history['train_mae'].append(train_mae/n_train)
        history['val_loss'].append(val_loss/n_val)
        history['val_mae'].append(val_mae/n_val)
        print(f"Epoch {epoch+1}: TrainLoss={train_loss/n_train:.4f}, ValLoss={val_loss/n_val:.4f}")
    return history

def plot_history(history, path):
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.plot(history["train_mae"], label="Train MAE", linestyle="--")
    plt.plot(history["val_mae"], label="Val MAE", linestyle="--")
    plt.legend(); plt.grid(); plt.tight_layout()
    plt.savefig(path); plt.close()

# ---------------- Main ----------------
def run_cross_validation():
    base_path = Path("/Users/jylee/Desktop/motor predict/model_1_final_asd")
    df_r = pd.read_csv(base_path.parent / "data/raw_data_robust_downsampled.csv")
    df_s = pd.read_csv(base_path.parent / "data/raw_data_soft_downsampled.csv")
    norm_df = pd.read_excel(base_path.parent / "data/normalization_verification.xlsx")
    norm_dict = {
        row['profile_id']: {col: row[col] for col in ['i_d', 'i_q', 'u_d', 'u_q', 'pm']}
        for _, row in norm_df.iterrows()
    }

    feature_cols = ['u_q', 'u_d', 'i_q', 'i_d']
    target_col = 'pm'
    look_back = 100
    tgt_len = 50
    input_dim = len(feature_cols)
    hidden_dim = 64
    num_layers = 2
    output_dim = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = apply_mixed_feature_normalization(df_r, df_s, norm_dict, feature_cols, target_col)

    fold_info = {
    "Subset_1": [
        {"train": [5, 6, 12, 13, 15, 21, 26, 31, 41, 42, 47, 48, 50, 66, 70, 74, 79],
         "val":   [49, 56, 57, 61, 63, 78],
         "test":  [29, 36, 58, 59, 69]},
    ],
    "Subset_2": [
        {"train": [4, 6, 8, 10, 11, 12, 14, 15, 17, 26, 27, 30, 52, 66, 71, 75, 78],
         "val":   [48, 59, 61, 63, 65, 76],
         "test":  [3, 53, 54, 80, 81]},
    ],
    "Subset_3": [
        {"train": [3, 4, 5, 11, 15, 16, 17, 20, 26, 27, 32, 43, 50, 58, 64, 79],
         "val":   [9, 45, 53, 55, 60, 63],
         "test":  [6, 13, 48, 59, 68, 72]},
    ],
    "Subset_4": [
        {"train": [8, 9, 15, 16, 20, 31, 43, 44, 47, 51, 52, 53, 59, 61, 63, 64, 80],
         "val":   [4, 12, 46, 50, 69, 79],
         "test":  [10, 29, 36, 74, 78]}
    ],
    "Subset_5": [
        {"train": [4, 6, 10, 14, 15, 17, 23, 45, 49, 50, 53, 56, 58, 74, 76, 78, 80],
         "val":   [57, 61, 65, 66, 68, 79],
         "test":  [24, 26, 42, 63, 72]},
    ]
}

    for subset_name, folds in fold_info.items():
        for fold_idx, fold_dict in enumerate(folds, 1):
            train_ids = fold_dict["train"]
            val_ids = fold_dict["val"]
            test_ids = fold_dict["test"]

            X_train, y_train, d_train = make_dataset_seq2seq(df, feature_cols, target_col, look_back, tgt_len, train_ids)
            X_val, y_val, d_val = make_dataset_seq2seq(df, feature_cols, target_col, look_back, tgt_len, val_ids)

            train_ds = TimeSeriesSeq2SeqDataset(X_train, y_train, d_train)
            val_ds = TimeSeriesSeq2SeqDataset(X_val, y_val, d_val)
            train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
            val_dl = DataLoader(val_ds, batch_size=32, shuffle=False)

            encoder = Encoder(input_dim, hidden_dim, num_layers).to(device)
            decoder = Decoder(output_dim, hidden_dim, num_layers).to(device)
            model = Seq2Seq(encoder, decoder).to(device)

            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            save_dir = base_path / f"model_1/{subset_name}/fold_{fold_idx}"
            os.makedirs(save_dir, exist_ok=True)

            print(f"\n[Training] {subset_name} - Fold {fold_idx} ...")
            history = train_model(model, train_dl, val_dl, criterion, optimizer, device, tgt_len)

            torch.save(model.state_dict(), save_dir / "model.pt")
            plot_history(history, save_dir / "loss_graph.png")
            with open(save_dir / "log.txt", "w") as f:
                json.dump(history, f, indent=2)
            with open(save_dir / "info.txt", "w") as f:
                f.write(f"Subset IDs: {train_ids + val_ids + test_ids}\n")
                f.write(f"Train: {train_ids}\nVal: {val_ids}\nTest: {test_ids}")

if __name__ == "__main__":
    run_cross_validation()
