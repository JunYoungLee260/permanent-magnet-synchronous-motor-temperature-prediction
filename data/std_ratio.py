import pandas as pd
import numpy as np

# ✅ 설정
input_path  = "/Users/jylee/Desktop/motor predict/data/raw_data.csv"
output_path = "/Users/jylee/Desktop/motor predict/data/std_ratios.csv"
EPS = 1e-12  # iqr=0, pm_std=0 보호용

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

# ✅ 비율 계산 (입력 std / pm std)
ratios_df = std_df.copy()
for col in ['i_d','i_q','u_d','u_q']:
    ratios_df[f"{col}_to_pm_ratio"] = ratios_df[col] / ratios_df['pm'].replace(0, EPS)

# ✅ 최종 컬럼 정리: profile_id + ratio 4개
output_df = ratios_df[['profile_id','i_d_to_pm_ratio','i_q_to_pm_ratio','u_d_to_pm_ratio','u_q_to_pm_ratio']]

# ✅ 저장
output_df.to_csv(output_path, index=False)
print(f"✅ 전체 id std 비율 결과 저장 완료 → {output_path}")
