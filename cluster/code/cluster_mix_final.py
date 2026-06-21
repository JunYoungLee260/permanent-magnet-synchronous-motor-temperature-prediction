import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from fastdtw import fastdtw

# --- apply_mixed_feature_normalization 함수 정의 (요청에 따라 추가) ---
def apply_mixed_feature_normalization(df_r, df_s, norm_dict, feature_cols, target_col):
    dfs_combined = []
    profile_ids = sorted(set(df_r['profile_id'].unique()) | set(df_s['profile_id'].unique()))
    for pid in profile_ids:
        row_r = df_r[df_r['profile_id'] == pid].copy().reset_index(drop=True)
        row_s = df_s[df_s['profile_id'] == pid].copy().reset_index(drop=True)
        mixed = pd.DataFrame()
        mixed['profile_id'] = row_r['profile_id']
        for col in feature_cols + [target_col]:
            # norm_dict에서 해당 컬럼의 정규화 방식을 가져옴
            source = norm_dict[pid].get(col)
            if source == "soft":
                mixed[col] = row_s[col]
            else:
                mixed[col] = row_r[col]
        dfs_combined.append(mixed)
    return pd.concat(dfs_combined).reset_index(drop=True)

# ------------------- 파일 경로 및 데이터 로딩 (수정됨) -------------------
# 이 부분은 실제 파일 경로에 맞게 수정해야 합니다.
base_path = "/Users/jylee/Desktop/motor predict"
robust_path = os.path.join(base_path, "data/raw_data_robust_downsampled.csv")
soft_path   = os.path.join(base_path, "data/raw_data_soft_downsampled.csv")
norm_path   = os.path.join(base_path, "data/normalization_verification.xlsx")

# 필요한 데이터 파일들을 로드
df_r = pd.read_csv(robust_path)
df_s = pd.read_csv(soft_path)
norm_df = pd.read_excel(norm_path) # normalization_verification 파일은 xlsx 형식
norm_dict = {
    row['profile_id']: {col: row[col] for col in ['i_d', 'i_q', 'u_d', 'u_q', 'pm']}
    for _, row in norm_df.iterrows()
}
feature_cols = ['u_q', 'u_d', 'i_q', 'i_d']
target_col   = 'pm'

# ------------------- 정규화 혼합 방식 적용 (새로운 코드) -------------------
# 두 데이터셋과 정규화 방식을 혼합하여 최종 데이터프레임 생성
df_mixed = apply_mixed_feature_normalization(df_r, df_s, norm_dict, feature_cols, target_col)

# ------------------- PM 데이터 추출 및 필터링 (수정됨) -------------------
pm_series_dict = {}
pids = []
# 혼합 정규화된 데이터프레임에서 PM 시계열 데이터를 추출
for pid in sorted(df_mixed['profile_id'].unique()):
    pm = df_mixed[df_mixed['profile_id'] == pid]['pm'].values
    # 다운샘플링은 이미 완료된 데이터라고 가정하고, 길이 필터만 적용
    if len(pm) > 1:
        pm_series_dict[pid] = pm
        pids.append(pid)

# ------------------- FastDTW 거리 행렬 계산 (이전 코드 유지) -------------------
n = len(pids)
dist_matrix = np.zeros((n, n))
scalar_distance = lambda u, v: abs(u - v)
for i in range(n):
    for j in range(i + 1, n):
        dist, _ = fastdtw(pm_series_dict[pids[i]], pm_series_dict[pids[j]], dist=scalar_distance)
        dist_matrix[i, j] = dist
        dist_matrix[j, i] = dist

# ------------------- 군집 수 8로 고정 (이전 코드 유지) -------------------
linked = linkage(squareform(dist_matrix), method='ward')
final_labels = fcluster(linked, t=8, criterion='maxclust')
cluster_df = pd.DataFrame({'profile_id': pids, 'cluster': final_labels})
cluster_df.to_csv(os.path.join(base_path, "cluster/cluster_result.csv"), index=False)

# ------------------- 덴드로그램 (이전 코드 유지) -------------------
plt.figure(figsize=(12, 6))
dendrogram(linked, labels=pids, orientation='top')
plt.title("Hierarchical Clustering of PM Time Series (FastDTW)")
plt.xlabel("Profile ID")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig(os.path.join(base_path, "cluster/dendrogram.png"))
plt.close()

# ------------------- 클러스터별 ID를 엑셀 파일로 저장 -------------------
cluster_id_map = {c: [] for c in sorted(cluster_df['cluster'].unique())}
for _, row in cluster_df.iterrows():
    cluster_id_map[row['cluster']].append(row['profile_id'])

# 딕셔너리를 데이터프레임으로 변환
# 클러스터 ID를 열 이름으로, ID 목록을 값으로 저장
df_to_save = pd.DataFrame({f"Cluster_{c}": pd.Series(ids) for c, ids in cluster_id_map.items()})

# Excel 파일로 저장
excel_save_path = os.path.join(base_path, "cluster/cluster_id_map.xlsx")
df_to_save.to_excel(excel_save_path, index=False)
print(f"\n✅ 클러스터별 ID 목록이 '{excel_save_path}'에 저장되었습니다.")

# ------------------- 군집별 시각화 (이전 코드 유지) -------------------
for c in sorted(cluster_df['cluster'].unique()):
    ids_in_cluster = cluster_df[cluster_df['cluster'] == c]['profile_id']
    plt.figure(figsize=(12, 4))
    for pid in ids_in_cluster:
        plt.plot(pm_series_dict[pid], label=f"ID {pid}", linewidth=0.8)
    plt.title(f"Cluster {c} - PM Time Series (Normalized)")
    plt.xlabel("Time Index")
    plt.ylabel("PM")
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, f"cluster/cluster_{c}.png"))
    plt.close()