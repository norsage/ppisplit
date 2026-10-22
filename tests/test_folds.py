from ppisplit.folds import assign_kfold, assign_montecarlo, assign_montecarlo_balanced


def test_kfold_each_uid_appears_once():
    clusters = {0: ["a", "b"], 1: ["c"], 2: ["d", "e"], 3: ["f"]}
    df = assign_kfold(clusters, n_folds=3, seed=42)
    assert sorted(df["uid"]) == ["a", "b", "c", "d", "e", "f"]


def test_kfold_each_uid_is_val_in_exactly_one_fold():
    clusters = {0: ["a", "b"], 1: ["c"], 2: ["d", "e"], 3: ["f"]}
    df = assign_kfold(clusters, n_folds=3, seed=42)
    val_counts = (df.drop(columns="uid") == "val").sum(axis=1)
    assert (val_counts == 1).all()


def test_kfold_cluster_atomicity():
    # All UIDs in a cluster must land in the same fold.
    clusters = {0: ["a", "b", "c"], 1: ["d"]}
    df = assign_kfold(clusters, n_folds=2, seed=42)
    # Find which fold has 'a' as val; b and c must agree.
    a_row = df[df["uid"] == "a"].iloc[0]
    b_row = df[df["uid"] == "b"].iloc[0]
    c_row = df[df["uid"] == "c"].iloc[0]
    for f in range(2):
        assert a_row[f"split_{f}"] == b_row[f"split_{f}"] == c_row[f"split_{f}"]


def test_kfold_balanced_assignment():
    # 6 equal-size clusters, 3 folds -> each fold should get exactly 2.
    clusters = {i: [f"u{i}_{j}" for j in range(2)] for i in range(6)}
    df = assign_kfold(clusters, n_folds=3, seed=7)
    val_counts_per_fold = [(df[f"split_{f}"] == "val").sum() for f in range(3)]
    assert val_counts_per_fold == [4, 4, 4]


def test_kfold_rejects_n_folds_below_2():
    import pytest
    with pytest.raises(ValueError):
        assign_kfold({0: ["a"]}, n_folds=1, seed=0)


def test_montecarlo_split_count_matches_n_splits():
    clusters = {i: [f"c{i}_{j}" for j in range(3)] for i in range(10)}
    df = assign_montecarlo(clusters, n_splits=5, val_ratio=0.25, seed=42)
    split_cols = [c for c in df.columns if c.startswith("split_")]
    assert len(split_cols) == 5


def test_montecarlo_val_ratio_approximate():
    clusters = {i: [f"c{i}_{j}" for j in range(2)] for i in range(50)}  # 100 UIDs total
    df = assign_montecarlo(clusters, n_splits=4, val_ratio=0.25, seed=42)
    for f in range(4):
        n_val = (df[f"split_{f}"] == "val").sum()
        # Cluster atomicity makes exact 0.25 unlikely; tolerate +/- one cluster (2 UIDs).
        assert 24 <= n_val <= 28


def test_montecarlo_cluster_atomicity():
    clusters = {0: ["a", "b", "c"], 1: ["d", "e"], 2: ["f"]}
    df = assign_montecarlo(clusters, n_splits=10, val_ratio=0.5, seed=42)
    # For every split, all UIDs in the same cluster share the same label.
    for f in range(10):
        col = f"split_{f}"
        labels_by_cluster = {
            0: {df.loc[df["uid"] == u, col].iloc[0] for u in clusters[0]},
            1: {df.loc[df["uid"] == u, col].iloc[0] for u in clusters[1]},
            2: {df.loc[df["uid"] == u, col].iloc[0] for u in clusters[2]},
        }
        for label_set in labels_by_cluster.values():
            assert len(label_set) == 1


def test_montecarlo_rejects_invalid_val_ratio():
    import pytest
    with pytest.raises(ValueError):
        assign_montecarlo({0: ["a"]}, n_splits=1, val_ratio=0.0, seed=0)
    with pytest.raises(ValueError):
        assign_montecarlo({0: ["a"]}, n_splits=1, val_ratio=1.0, seed=0)


def test_montecarlo_reproducible_with_seed():
    clusters = {i: [f"c{i}_{j}" for j in range(2)] for i in range(20)}
    df1 = assign_montecarlo(clusters, n_splits=3, val_ratio=0.3, seed=99)
    df2 = assign_montecarlo(clusters, n_splits=3, val_ratio=0.3, seed=99)
    assert df1.equals(df2)


def _skewed_clusters() -> dict[int, list[str]]:
    # One cluster of 60, three of 20 (21% of the UIDs), and 180 small ones of 1-4: 120 + 450 = 570 UIDs.
    sizes = [60, 20, 20, 20] + [1 + i % 4 for i in range(180)]
    return {cid: [f"c{cid}_{j}" for j in range(n)] for cid, n in enumerate(sizes)}


def test_montecarlo_balanced_val_size_within_one_small_cluster():
    clusters = _skewed_clusters()
    total = sum(len(u) for u in clusters.values())
    df = assign_montecarlo_balanced(clusters, n_splits=200, val_ratio=0.25, seed=0, large_threshold=10)
    for f in range(200):
        n_val = (df[f"split_{f}"] == "val").sum()
        assert total * 0.25 - 4 < n_val <= total * 0.25


def test_montecarlo_balanced_every_cluster_val_at_val_ratio():
    # Large and small clusters alike land in val with probability ~ val_ratio.
    clusters = _skewed_clusters()
    n = 2000
    df = assign_montecarlo_balanced(clusters, n_splits=n, val_ratio=0.25, seed=1, large_threshold=10)
    is_val = (df.set_index("uid") == "val").to_numpy()
    row = {uid: i for i, uid in enumerate(df["uid"])}
    freq = {cid: is_val[row[uids[0]]].mean() for cid, uids in clusters.items()}
    # Binomial sd at p=0.25, n=2000 is ~0.0097; skipping near the end lowers the larger small clusters a little.
    for cid in range(4):
        assert abs(freq[cid] - 0.25) < 0.04, (cid, freq[cid])
    for k in (1, 2, 3, 4):
        group = [f for cid, f in freq.items() if cid >= 4 and len(clusters[cid]) == k]
        assert abs(sum(group) / len(group) - 0.25) < 0.02, (k, sum(group) / len(group))


def test_montecarlo_balanced_weights_set_the_size():
    # Two uids; "a" weighs 3 times "b". With val_ratio 0.25 of weight 4, only "b" fits.
    clusters = {0: ["a"], 1: ["b"]}
    df = assign_montecarlo_balanced(
        clusters, n_splits=20, val_ratio=0.25, seed=0, large_threshold=10, weights={"a": 3, "b": 1}
    )
    assert (df.set_index("uid").loc["a"] == "train").all()
    assert (df.set_index("uid").loc["b"] == "val").all()


def test_montecarlo_balanced_cluster_atomicity():
    clusters = _skewed_clusters()
    df = assign_montecarlo_balanced(clusters, n_splits=20, val_ratio=0.25, seed=3, large_threshold=10)
    labels = df.set_index("uid")
    for uids in clusters.values():
        assert (labels.loc[uids].nunique() == 1).all()


def test_montecarlo_balanced_rejects_large_clusters_above_val_ratio():
    import pytest
    clusters = {0: [f"a{i}" for i in range(50)], 1: [f"b{i}" for i in range(50)]}
    with pytest.raises(ValueError, match="large_threshold"):
        assign_montecarlo_balanced(clusters, n_splits=1, val_ratio=0.25, seed=0, large_threshold=10)


def test_montecarlo_balanced_rejects_missing_weight():
    import pytest
    with pytest.raises(ValueError, match="no weight"):
        assign_montecarlo_balanced({0: ["a"], 1: ["b"]}, n_splits=1, weights={"a": 1.0})
