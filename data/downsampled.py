import pandas as pd

# 파일 경로 설정
robust_path = "/Users/jylee/Desktop/motor predict/data/raw_data.csv"


# 저장 경로 설정
robust_out_path = "/Users/jylee/Desktop/motor predict/data/raw_data_downsampled.csv"


# 데이터 로딩
df_robust = pd.read_csv(robust_path)


# 다운샘플링 (4개마다 1개씩 추출 → 1/4 크기)
df_robust_down = df_robust.iloc[::4].reset_index(drop=True)


# 저장
df_robust_down.to_csv(robust_out_path, index=False)


print("✅ 다운샘플링 완료")
print(f"🔹 robust: {df_robust.shape} → {df_robust_down.shape}")

