import pandas as pd
import numpy as np

# 🔧 경로 설정
input_path = "/Users/jylee/Desktop/motor predict/data/raw_data_robust_only.csv"
output_path = "/Users/jylee/Desktop/motor predict/data/raw_data_soft_scaled.csv"
soft_std_path = "/Users/jylee/Desktop/motor predict/data/soft_std.csv"

# 1. 데이터 불러오기
df = pd.read_csv(input_path)

# 2. 적용할 컬럼
features = ["i_d", "i_q", "u_d", "u_q", "pm"]

# 3. soft_std 계산 함수
def compute_soft_std(group):
    all_values = pd.concat([group[col] for col in features], axis=0)
    return np.sqrt(np.var(all_values))

# 4. profile_id별 soft_std 계산
soft_std_series = df.groupby("profile_id").apply(compute_soft_std).rename("soft_std")
soft_std_df = soft_std_series.reset_index()
soft_std_df.to_csv(soft_std_path, index=False)
print(f"✅ soft_std 저장 완료 → {soft_std_path}")

# 5. soft_std를 df에 merge
df = df.merge(soft_std_df, on="profile_id")

# 6. soft rescaling 적용
for col in features:
    df[col] = df[col] / df["soft_std"]

# 7. 저장 (컬럼명 그대로 유지)
df.drop(columns=["soft_std"]).to_csv(output_path, index=False)
print(f"✅ soft scaling 완료 → {output_path}")
