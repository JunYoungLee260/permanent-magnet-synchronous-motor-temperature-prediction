import numpy as np
from sklearn.model_selection import KFold

# 1. 클러스터 수동 정의
cluster_id_map = {
    1: [7, 9, 12, 14, 15, 16, 18, 19, 24, 26, 70],
    2: [17, 45, 5, 61, 21, 56, 2, 62, 3, 43, 23, 47, 10, 67, 13, 52, 8],
    3: [4, 29, 6],
    4: [76, 44, 48, 79],
    5: [32, 74, 27, 65, 31, 66, 20],
    6: [41, 55, 80, 63],
    7: [30, 53, 36, 58],
    8: [11, 69, 75, 64, 68, 72, 50, 73, 49, 42, 81, 60, 59, 46, 54, 78, 57, 71, 51]
}

total_ids_to_select = 36
num_subsets = 5
num_folds = 5
val_n = 7  # validation 개수 고정

print(f"\n📌 총 {num_subsets}개의 서브셋에 대해 {num_folds}-Fold Cross Validation을 생성합니다.\n")

for subset_index in range(num_subsets):
    # 2. 서브셋 샘플링
    np.random.seed(subset_index)
    min_per_cluster = {cid: 1 for cid in cluster_id_map}
    cluster_sizes = {cid: len(ids) for cid, ids in cluster_id_map.items()}
    residual = total_ids_to_select - sum(min_per_cluster.values())
    total_size = sum(cluster_sizes.values())
    weights = {cid: size / total_size for cid, size in cluster_sizes.items()}
    extra_alloc = {cid: int(round(weights[cid] * residual)) for cid in cluster_id_map}

    while sum(min_per_cluster[cid] + extra_alloc[cid] for cid in cluster_id_map) > total_ids_to_select:
        target = max(extra_alloc, key=extra_alloc.get)
        if extra_alloc[target] > 0:
            extra_alloc[target] -= 1
    while sum(min_per_cluster[cid] + extra_alloc[cid] for cid in cluster_id_map) < total_ids_to_select:
        target = min(extra_alloc, key=extra_alloc.get)
        extra_alloc[target] += 1

    final_alloc = {
        cid: min(min_per_cluster[cid] + extra_alloc[cid], len(cluster_id_map[cid]))
        for cid in cluster_id_map
    }

    subset_ids = []
    for cid, ids in cluster_id_map.items():
        k = final_alloc[cid]
        sampled = np.random.choice(ids, size=k, replace=False).tolist()
        subset_ids.extend(sampled)
    subset_ids = np.array(sorted(subset_ids))

    print(f"📦 Subset {subset_index + 1}: {subset_ids.tolist()}")

    # 3. 5-Fold Cross Validation
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=subset_index)
    for fold_index, (train_val_idx, test_idx) in enumerate(kf.split(subset_ids)):
        train_val_ids = subset_ids[train_val_idx]
        test_ids = subset_ids[test_idx]

        np.random.seed(fold_index)
        val_indices = np.random.choice(len(train_val_ids), size=val_n, replace=False)
        val_ids = train_val_ids[val_indices]
        train_ids = np.delete(train_val_ids, val_indices)

        print(f"  🔁 Fold {fold_index + 1}")
        print(f"    Train ({len(train_ids)}): {sorted(train_ids.tolist())}")
        print(f"    Val   ({len(val_ids)}): {sorted(val_ids.tolist())}")
        print(f"    Test  ({len(test_ids)}): {sorted(test_ids.tolist())}")
    print("-" * 80)
