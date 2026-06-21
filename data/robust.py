import pandas as pd

# 🔧 파일 경로
input_path = "/Users/jylee/Desktop/motor predict/data/raw_data_filtered.csv"
output_scaled_path = "/Users/jylee/Desktop/motor predict/data/raw_data_robust_only.csv"
output_pm_stats_path = "/Users/jylee/Desktop/motor predict/data/pm_scaler_stats.csv"

# 1. 데이터 불러오기
df = pd.read_csv(input_path)

# 2. Robust Scaling (각 feature별, profile_id 기준, 컬럼명 그대로 덮어씀)
def robust_scale_column(df, column):
    grouped = df.groupby("profile_id")[column]
    median = grouped.transform("median")
    q1 = grouped.transform(lambda x: x.quantile(0.25))
    q3 = grouped.transform(lambda x: x.quantile(0.75))
    iqr = q3 - q1
    scaled = (df[column] - median) / iqr.replace(0, 1e-9)  # iqr이 0인 경우 대비
    return scaled

# 3. 정규화 적용할 컬럼 목록
cols = ["i_d", "i_q", "u_d", "u_q", "pm"]

# 4. 각 열에 대해 정규화 후 원본을 덮어쓰기
for col in cols:
    df[col] = robust_scale_column(df, col)

# 5. 정규화된 값 + profile_id만 저장
df[["profile_id"] + cols].to_csv(output_scaled_path, index=False)
print(f"✅ robust scaling 적용 완료 → {output_scaled_path}")

# 6. pm의 profile_id별 median, IQR 계산 후 저장
df_raw = pd.read_csv(input_path, usecols=["profile_id", "pm"])  # 원본 pm으로부터 계산
pm_stats = df_raw.groupby("profile_id")["pm"].agg(
    median="median",
    q1=lambda x: x.quantile(0.25),
    q3=lambda x: x.quantile(0.75)
)
pm_stats["iqr"] = pm_stats["q3"] - pm_stats["q1"]
pm_stats = pm_stats.reset_index()[["profile_id", "median", "iqr"]]
pm_stats.to_csv(output_pm_stats_path, index=False)
print(f"✅ pm 기준값 저장 완료 → {output_pm_stats_path}")
