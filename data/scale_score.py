import pandas as pd
import numpy as np

# ✅ 설정
input_path  = "/Users/jylee/Desktop/motor predict/data/raw_data.csv"
EPS = 1e-12  # iqr=0, pm_std=0 보호용
SELECTED_IDS = [2]   # 확인하고 싶은 profile_id 리스트 (예: [17] 또는 [17,29])

# ✅ 데이터 로드
df = pd.read_csv(input_path)
cols = ['i_d', 'i_q', 'u_d', 'u_q', 'pm', 'profile_id']
df = df[cols]

# ✅ robust 정규화
def robust_normalize(group):
    result = group.copy()
    for col in ['i_d', 'i_q', 'u_d', 'u_q', 'pm']:
        med = group[col].median()
        iqr = group[col].quantile(0.75) - group[col].quantile(0.25)
        denom = iqr if np.isfinite(iqr) and iqr != 0 else EPS
        result[col] = (group[col] - med) / denom
    return result

df_scaled = df.groupby("profile_id", group_keys=False).apply(robust_normalize)

# ✅ id 단위 std 집계
std_df = df_scaled.groupby("profile_id")[['i_d','i_q','u_d','u_q','pm']].std(ddof=0).reset_index()

# ✅ 특정 id만 콘솔 출력
for pid in SELECTED_IDS:
    row = std_df[std_df['profile_id'] == pid]
    if row.empty:
        print(f"[ID {pid}] 데이터 없음.")
        continue

    r = row.iloc[0]
    pm_std = float(r['pm']) if np.isfinite(r['pm']) else 0.0
    pm_std_safe = pm_std if pm_std > 0 else EPS

    print(f"\n===== profile_id: {pid} =====")
    print(f"pm_std : {pm_std:.6g}")
    for col in ['i_d','i_q','u_d','u_q']:
        ratio = float(r[col]) / pm_std_safe
        print(f"{col}_std : {float(r[col]):.6g}  |  ratio_to_pm : {ratio:.6g}")
