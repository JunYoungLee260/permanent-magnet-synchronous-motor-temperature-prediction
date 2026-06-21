import os, json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

# ---------------- Dataset ----------------
class TimeSeriesSeq2SeqDataset(Dataset):
    def __init__(self, X, y, decoder_starts):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)          # (B, T, 1)
        self.decoder_starts = torch.tensor(decoder_starts, dtype=torch.float32)\
                               .unsqueeze(-1).unsqueeze(-1)                  # (B, 1, 1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx], self.decoder_starts[idx]

# ---------------- Additive Attention (Bahdanau) ----------------
class AdditiveAttention(nn.Module):
    """
    score(s_t, h_i) = v^T tanh(W_q s_t + W_k h_i)
    s_t: (B, H)   (decoder state)
    h_i: (B, S, H) (encoder outputs)
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v   = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, query, encoder_outputs):
        # query: (B, H)
        # encoder_outputs: (B, S, H)
        B, S, H = encoder_outputs.size()
        q = self.W_q(query).unsqueeze(1).expand(-1, S, -1)    # (B, S, H)
        k = self.W_k(encoder_outputs)                         # (B, S, H)
        energy = torch.tanh(q + k)                            # (B, S, H)
        scores = self.v(energy).squeeze(-1)                   # (B, S)
        attn_weights = torch.softmax(scores, dim=1)           # (B, S)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (B, 1, H)
        return context, attn_weights

# ---------------- Encoder (2-layer LSTM) ----------------
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        # outputs: (B, S, H); hidden/cell: (num_layers, B, H)
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, hidden, cell

# ---------------- Decoder with 2-layer Additive Attention ----------------
class DecoderAdditive2L(nn.Module):
    """
    두 층 각각에 Bahdanau attention.
    - 1층 입력: concat([y_{t-1}, context1])  -> LSTMCell1
    - 2층 입력: concat([h1_t,   context2])  -> LSTMCell2
    - 출력: fc(h2_t) -> (B, 1)
    """
    def __init__(self, output_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 두 개의 LSTMCell (한 타임스텝씩 전개)
        self.lstm1 = nn.LSTMCell(input_size=1 + hidden_dim, hidden_size=hidden_dim)
        self.lstm2 = nn.LSTMCell(input_size=hidden_dim + hidden_dim, hidden_size=hidden_dim)

        # 두 층용 Additive Attention
        self.attn1 = AdditiveAttention(hidden_dim)
        self.attn2 = AdditiveAttention(hidden_dim)

        # 최종 예측
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)   # output_dim = 1
        )

    def forward(self, y_prev, h1, c1, h2, c2, encoder_outputs):
        """
        y_prev: (B, 1, 1)  디코더 입력 스칼라
        h1,c1,h2,c2: (B, H)
        encoder_outputs: (B, S, H)
        """
        B = y_prev.size(0)
        y_prev_s = y_prev.squeeze(1).squeeze(1)        # (B,)

        # --- 1층 어텐션 & LSTMCell(1)
        ctx1, _ = self.attn1(h1, encoder_outputs)      # ctx1: (B, 1, H)
        x1 = torch.cat([y_prev, ctx1], dim=2).squeeze(1)   # (B, 1+H)
        h1, c1 = self.lstm1(x1, (h1, c1))              # (B, H)

        # --- 2층 어텐션 & LSTMCell(2)
        ctx2, _ = self.attn2(h2, encoder_outputs)      # (B, 1, H)
        x2 = torch.cat([h1.unsqueeze(1), ctx2], dim=2).squeeze(1)  # (B, H+H)
        h2, c2 = self.lstm2(x2, (h2, c2))              # (B, H)

        # --- 출력
        y_hat = self.fc(h2).unsqueeze(1)               # (B, 1, 1)
        return y_hat, h1, c1, h2, c2

# ---------------- Seq2Seq Wrapper ----------------
class Seq2SeqAdditive2L(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, decoder_init, tgt_len, y=None, teacher_forcing_ratio=0.5):
        # 인코더
        encoder_outputs, hidden, cell = self.encoder(src)  # hidden/cell: (2, B, H)
        h1, h2 = hidden[0], hidden[1]                      # (B, H)
        c1, c2 = cell[0], cell[1]                          # (B, H)

        outputs = []
        y_prev = decoder_init                               # (B,1,1)
        for t in range(tgt_len):
            y_hat, h1, c1, h2, c2 = self.decoder(y_prev, h1, c1, h2, c2, encoder_outputs)
            outputs.append(y_hat)

            # 확률적 티쳐포싱 (모델2와 동일, per-step)
            if y is not None and torch.rand(1).item() < teacher_forcing_ratio:
                y_prev = y[:, t:t+1, :]                    # GT
            else:
                y_prev = y_hat.detach()                    # 자기 예측

        return torch.cat(outputs, dim=1)                   # (B, T, 1)

# ---------------- Data utils ----------------
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

# ---------------- Train/Eval ----------------
def train_model(model, train_loader, val_loader, criterion, optimizer, device, tgt_len, num_epochs=50):
    history = {"train_loss": [], "train_mae": [], "val_loss": [], "val_mae": []}
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]") 
        model.train()
        train_loss, train_mae = 0, 0
        # 모델2와 동일한 선형 감소 스케줄
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
            train_mae  += mae.item()

        model.eval()
        val_loss, val_mae = 0, 0
        with torch.no_grad():
            for x, y, dec in val_loader:
                x, y, dec = x.to(device), y.to(device), dec.to(device)
                pred = model(x, dec, tgt_len, y, 0.0)     # 검증은 teacher forcing 없이
                loss = criterion(pred, y)
                mae = torch.mean(torch.abs(pred - y))
                val_loss += loss.item()
                val_mae  += mae.item()

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

# ---------------- Main ----------------
def run_cross_validation(fold_info):
    base_path = Path("/Users/jylee/Desktop/motor predict/model_3_final_2att_69")
    robust_path = base_path.parent / "data/raw_data_robust_downsampled.csv"
    soft_path   = base_path.parent / "data/raw_data_soft_downsampled.csv"
    norm_path   = base_path.parent / "data/normalization_verification.xlsx"

    df_r = pd.read_csv(robust_path)
    df_s = pd.read_csv(soft_path)
    norm_df = pd.read_excel(norm_path)
    norm_dict = {
        row['profile_id']: {col: row[col] for col in ['i_d', 'i_q', 'u_d', 'u_q', 'pm']}
        for _, row in norm_df.iterrows()
    }

    feature_cols = ['u_q', 'u_d', 'i_q', 'i_d']
    target_col   = 'pm'
    look_back, tgt_len = 100, 50
    input_dim, hidden_dim, num_layers, output_dim = len(feature_cols), 64, 2, 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = apply_mixed_feature_normalization(df_r, df_s, norm_dict, feature_cols, target_col)

    for subset_name, folds in fold_info.items():
        for fold_idx, fold_dict in enumerate(folds, 1):
            train_ids, val_ids, test_ids = fold_dict["train"], fold_dict["val"], fold_dict["test"]

            X_train, y_train, d_train = make_dataset_seq2seq(df, feature_cols, target_col, look_back, tgt_len, train_ids)
            X_val,   y_val,   d_val   = make_dataset_seq2seq(df, feature_cols, target_col, look_back, tgt_len, val_ids)

            train_ds = TimeSeriesSeq2SeqDataset(X_train, y_train, d_train)
            val_ds   = TimeSeriesSeq2SeqDataset(X_val,   y_val,   d_val)
            train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
            val_dl   = DataLoader(val_ds,   batch_size=32, shuffle=False)

            encoder = Encoder(input_dim, hidden_dim, num_layers).to(device)
            decoder = DecoderAdditive2L(output_dim, hidden_dim).to(device)
            model   = Seq2SeqAdditive2L(encoder, decoder).to(device)

            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()

            # ⚠️ 비교를 위해 별도 폴더에 저장 (원하면 "model_2"로 변경)
            save_dir = base_path / f"model_2_additive_attn2/{subset_name}/fold_{fold_idx}"
            os.makedirs(save_dir, exist_ok=True)

            print(f"\n[Training] {subset_name} - Fold {fold_idx} ...")
            history = train_model(model, train_dl, val_dl, criterion, optimizer, device, tgt_len)

            torch.save(model.state_dict(), save_dir / "model.pt")
            plot_history(history, save_dir / "loss_graph.png")
            with open(save_dir / "log.txt", "w") as f:
                json.dump({k:[float(v) for v in vals] for k,vals in history.items()}, f, indent=2)
            with open(save_dir / "info.txt", "w") as f:
                f.write(f"Subset IDs: {train_ids + val_ids + test_ids}\n")
                f.write(f"Train: {train_ids}\nVal: {val_ids}\nTest: {test_ids}")

if __name__ == "__main__":
    fold_info = {
    "Subset_1": [
        {
            "train": [2, 3, 4, 5, 6, 9, 12, 13, 14, 16, 17, 19, 23, 24, 27, 29, 30, 31, 32, 36, 41, 43, 47, 48, 52, 53, 54, 55, 56, 58, 59, 64, 66, 67, 68, 69, 73, 75, 76, 78, 79],
            "val": [8, 10, 11, 18, 20, 44, 46, 50, 51, 57, 60, 71, 72, 81],
            "test": [7, 15, 21, 26, 42, 45, 49, 61, 62, 63, 65, 70, 74, 80]
        },
        {
            "train": [4, 5, 6, 7, 9, 11, 12, 13, 14, 18, 20, 21, 24, 26, 27, 30, 31, 41, 45, 46, 47, 48, 51, 52, 54, 55, 57, 58, 59, 63, 64, 66, 67, 68, 70, 71, 73, 74, 78, 79, 80],
            "val": [2, 16, 23, 32, 36, 49, 50, 56, 60, 61, 65, 72, 76, 81],
            "test": [3, 8, 10, 15, 17, 19, 29, 42, 43, 44, 53, 62, 69, 75]
        },
        {
            "train": [2, 5, 6, 7, 8, 10, 11, 14, 16, 17, 20, 21, 23, 26, 29, 31, 41, 42, 43, 45, 47, 48, 53, 54, 56, 57, 59, 62, 64, 65, 66, 67, 68, 69, 71, 72, 74, 75, 76, 80, 81],
            "val": [3, 19, 27, 30, 32, 36, 44, 46, 52, 60, 61, 63, 70, 78],
            "test": [4, 9, 12, 13, 15, 18, 24, 49, 50, 51, 55, 58, 73, 79]
        },
        {
            "train": [4, 5, 7, 8, 9, 12, 13, 15, 17, 19, 20, 21, 23, 24, 26, 27, 31, 32, 36, 41, 43, 44, 48, 49, 50, 51, 53, 54, 55, 56, 58, 62, 63, 64, 67, 68, 71, 73, 75, 76, 79],
            "val": [2, 6, 11, 29, 30, 45, 46, 52, 60, 61, 66, 69, 74, 78],
            "test": [3, 10, 14, 16, 18, 42, 47, 57, 59, 65, 70, 72, 80, 81]
        },
        {
            "train": [2, 4, 9, 10, 11, 12, 14, 17, 23, 24, 26, 29, 31, 32, 36, 41, 43, 44, 48, 49, 51, 52, 54, 55, 56, 57, 59, 64, 65, 66, 67, 68, 69, 70, 71, 74, 75, 76, 78, 80, 81],
            "val": [5, 8, 13, 16, 18, 19, 27, 45, 46, 53, 58, 61, 72, 79],
            "test": [3, 6, 7, 15, 20, 21, 30, 42, 47, 50, 60, 62, 63, 73]
        }
    ]
}
 # 외부에서 fold_info 임포트한다고 가정
run_cross_validation(fold_info)