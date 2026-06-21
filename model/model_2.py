import os
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

# ---------------- fold_info ----------------
fold_info = {
    "Subset_1": [
        {"train": [5, 13, 15, 26, 29, 31, 36, 41, 47, 50, 56, 61, 63, 69, 70, 74],
         "val":   [6, 48, 58, 59, 78, 79],
         "test":  [12, 21, 42, 49, 57, 66]},
        {"train": [5, 12, 13, 21, 26, 29, 31, 36, 41, 42, 47, 49, 57, 63, 78, 79],
         "val":   [15, 50, 58, 59, 66, 69],
         "test":  [6, 48, 56, 61, 70, 74]},
        {"train": [6, 12, 13, 21, 29, 42, 47, 48, 49, 50, 57, 59, 61, 63, 66, 74],
         "val":   [5, 36, 56, 58, 69, 70],
         "test":  [15, 26, 31, 41, 78, 79]},
        {"train": [12, 15, 26, 29, 31, 36, 41, 42, 49, 57, 58, 59, 61, 66, 69, 70, 79],
         "val":   [6, 21, 48, 56, 74, 78],
         "test":  [5, 13, 47, 50, 63]}
    ],
    "Subset_2": [
        {"train": [3, 6, 10, 11, 12, 14, 15, 17, 26, 30, 48, 61, 66, 71, 75, 78],
         "val":   [4, 27, 53, 54, 80, 81],
         "test":  [8, 52, 59, 63, 65, 76]},
        {"train": [3, 4, 8, 12, 14, 15, 17, 27, 30, 48, 52, 53, 59, 65, 80, 81],
         "val":   [11, 54, 61, 63, 76, 78],
         "test":  [6, 10, 26, 66, 71, 75]},
        {"train": [6, 8, 10, 11, 15, 26, 27, 30, 52, 53, 59, 65, 66, 71, 75, 81],
         "val":   [3, 17, 54, 63, 76, 80],
         "test":  [4, 12, 14, 48, 61, 78]},
        {"train": [4, 6, 10, 12, 14, 26, 48, 52, 54, 61, 63, 65, 66, 71, 75, 76, 81],
         "val":   [3, 8, 53, 59, 78, 80],
         "test":  [11, 15, 17, 27, 30]}
    ],
    "Subset_3": [
        {"train": [5, 9, 11, 13, 15, 16, 20, 26, 27, 45, 48, 59, 60, 63, 64, 68],
         "val":   [6, 32, 53, 58, 72, 79],
         "test":  [3, 4, 17, 43, 50, 55]},
        {"train": [4, 6, 13, 15, 16, 26, 32, 43, 45, 48, 53, 59, 60, 63, 64, 79],
         "val":   [3, 17, 50, 55, 68, 72],
         "test":  [5, 9, 11, 20, 27, 58]},
        {"train": [3, 4, 5, 6, 9, 11, 13, 16, 17, 20, 27, 32, 45, 59, 60, 68, 79],
         "val":   [43, 48, 50, 55, 58, 72],
         "test":  [15, 26, 53, 63, 64]},
        {"train": [4, 5, 9, 11, 13, 15, 17, 20, 27, 48, 50, 53, 55, 58, 59, 63, 72],
         "val":   [3, 6, 26, 43, 64, 68],
         "test":  [16, 32, 45, 60, 79]}
    ],
    "Subset_4": [
        {"train": [4, 9, 10, 12, 15, 16, 20, 29, 31, 43, 46, 61, 63, 64, 69, 74],
         "val":   [8, 36, 47, 59, 78, 80],
         "test":  [44, 50, 51, 52, 53, 79]},
        {"train": [4, 10, 15, 20, 29, 31, 36, 43, 44, 50, 51, 52, 59, 64, 79, 80],
         "val":   [16, 53, 61, 63, 74, 78],
         "test":  [8, 9, 12, 46, 47, 69]},
        {"train": [8, 9, 10, 12, 29, 43, 44, 46, 47, 50, 52, 59, 63, 69, 74, 80],
         "val":   [4, 36, 51, 53, 78, 79],
         "test":  [15, 16, 20, 31, 61, 64]},
        {"train": [8, 9, 10, 12, 15, 16, 20, 29, 31, 36, 44, 46, 50, 64, 69, 74, 79],
         "val":   [47, 51, 52, 53, 61, 78],
         "test":  [4, 43, 59, 63, 80]},
    ],
    "Subset_5": [
        {"train": [4, 10, 14, 15, 17, 23, 24, 26, 42, 50, 53, 63, 66, 68, 72, 74],
         "val":   [6, 45, 56, 57, 76, 79],
         "test":  [49, 58, 61, 65, 78, 80]},
        {"train": [6, 10, 15, 23, 24, 26, 42, 49, 50, 53, 56, 58, 63, 72, 79, 80],
         "val":   [17, 61, 65, 68, 74, 78],
         "test":  [4, 14, 45, 57, 66, 76]},
        {"train": [6, 14, 17, 24, 26, 45, 49, 56, 57, 58, 63, 66, 72, 74, 76, 80],
         "val":   [4, 42, 61, 65, 78, 79],
         "test":  [10, 15, 23, 50, 53, 68]},
        {"train": [10, 14, 23, 24, 26, 42, 45, 49, 53, 58, 61, 63, 65, 66, 68, 72, 80],
         "val":   [4, 15, 50, 57, 76, 78],
         "test":  [6, 17, 56, 74, 79]}
    ]
}

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

# ---------------- 모델 정의 ----------------
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

    def forward(self, src, decoder_init, tgt_len, y=None, teacher_forcing_ratio=0.5):
        hidden, cell = self.encoder(src)
        outputs = []
        decoder_input = decoder_init
        for t in range(tgt_len):
            out, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs.append(out)

            if y is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = y[:, t:t+1, :]
            else:
                decoder_input = out.detach()
        return torch.cat(outputs, dim=1)

# ---------------- 데이터셋 생성 ----------------
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

# ✅ ---------------- 정규화 혼합 적용 (norm_dict 기반) ----------------
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

# ---------------- 학습 루프 ----------------
def train_model(model, train_loader, val_loader, criterion, optimizer, device, tgt_len, num_epochs=50):
    history = {"train_loss": [], "train_mae": [], "val_loss": [], "val_mae": []}
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_mae = 0, 0
        teacher_forcing_ratio = max(0.0, 1.0 - (epoch / num_epochs))
        for x, y, dec in train_loader:
            x, y, dec = x.to(device), y.to(device), dec.to(device)
            optimizer.zero_grad()
            pred = model(x, dec, tgt_len, y, teacher_forcing_ratio)
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
                pred = model(x, dec, tgt_len, y, 0.0)
                loss = criterion(pred, y)
                mae = torch.mean(torch.abs(pred - y))
                val_loss += loss.item()
                val_mae += mae.item()
        n_train, n_val = len(train_loader), len(val_loader)
        history["train_loss"].append(train_loss / n_train)
        history["train_mae"].append(train_mae / n_train)
        history["val_loss"].append(val_loss / n_val)
        history["val_mae"].append(val_mae / n_val)
        print(f"Epoch {epoch+1}: TrainLoss={train_loss/n_train:.4f}, ValLoss={val_loss/n_val:.4f}, TF={teacher_forcing_ratio:.2f}")
    return history

def plot_history(history, path):
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.plot(history["train_mae"], label="Train MAE", linestyle="--")
    plt.plot(history["val_mae"], label="Val MAE", linestyle="--")
    plt.legend(); plt.grid(); plt.tight_layout()
    plt.savefig(path); plt.close()

# ✅ ---------------- 실행 ----------------
def run_cross_validation():
    base_path = Path("/Users/jylee/Desktop/motor predict/model_2_final")
    robust_path = base_path.parent / "data/raw_data_robust_downsampled.csv"
    soft_path = base_path.parent / "data/raw_data_soft_downsampled.csv"
    norm_path = base_path.parent / "data/normalization_verification.xlsx"

    df_r = pd.read_csv(robust_path)
    df_s = pd.read_csv(soft_path)
    norm_df = pd.read_excel(norm_path)
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

    for subset_name, folds in fold_info.items():
        for fold_idx, fold_dict in enumerate(folds, 1):
            train_ids = fold_dict["train"]
            val_ids = fold_dict["val"]
            test_ids = fold_dict["test"]

            print(f"\n[Training] {subset_name} - Fold {fold_idx} ...")
            print(f"   - Train IDs: {train_ids}")
            print(f"   - Val   IDs: {val_ids}")
            print(f"   - Test  IDs: {test_ids}")

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

            save_dir = base_path / f"model_2/{subset_name}/fold_{fold_idx}"
            os.makedirs(save_dir, exist_ok=True)

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
