"""Command-line interface for ppisplit.

Subcommands:
    cluster      run mmseqs2 search and store the hit table
    split        derive complex clusters and assign CV folds
    check-leaks  identify (test_uid, train_uid) sequence-similarity leaks
    summary      cluster size statistics

The predicate flags configure the *post-filter*, not mmseqs's search; the search
itself runs at fixed recall settings that are chosen so they can never decide a
verdict (see ppisplit.search.SEARCH_SETTINGS).

Canonical protocol (the defaults below): a pair of chains is similar when
``fident * max(qcov, tcov) >= 0.24`` (``--predicate hyperbola --hyperbola-x 0.24
--cov-mode -1``), and a test complex leaks when **both** partners match the same
train complex (``--similarity-mode both``). The same predicate is written to
params.json by 'cluster' and re-read by 'split' / 'check-leaks' / 'clean-test-set',
so CV fold boundaries and test-leak verdicts are governed by one rule.

**Purpose-split scope.** One (query, target) database pair per question; never
pool. ``cluster --csv TEST --target-csv TRAIN`` builds the test-vs-train
comparison; ``cluster --csv TRAIN`` alone builds the self-search used for fold
clustering (and, with a test set as --csv, for within-set redundancy). Adding a
test set creates a new comparison, so it can never move an existing one's
verdicts. The old "focused universe" rule is obsolete — it was a workaround for
search parameters that scaled with database size.

The legacy rectangle predicate (0.3 identity / 0.8 coverage) is retained only to
reproduce historical numbers, as is ``--sequence-source seqres``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ppisplit.folds import assign_kfold, assign_montecarlo, assign_montecarlo_balanced
from ppisplit.graph import (
    chain_clusters_from_edges,
    complex_clusters_from_edges,
    exclude_test_clusters,
    group_clusters,
)
from ppisplit.io import (
    DEFAULT_SEQUENCE_SOURCE,
    SEQUENCE_SOURCES,
    chain_role_map,
    load_annotations,
    read_fasta_keys,
    write_fasta,
)
from ppisplit.leaks import detect_leaks
from ppisplit.predicates import (
    COV_MODE_SYMMETRIC,
    EdgePredicate,
    HyperbolaPredicate,
    RectanglePredicate,
)
from ppisplit.search import (
    SEARCH_SETTINGS,
    run_mmseqs_search_allvall,
    run_mmseqs_search_pair,
)


HITS_FILENAME = "hits.parquet"
PARAMS_FILENAME = "params.json"
FASTA_FILENAME = "sequences.fasta"
TARGET_FASTA_FILENAME = "sequences_target.fasta"


def require_known_chains(complexes, known_chains: set[str], command: str) -> None:
    """Exit if not one chain of ``complexes`` is in the cluster directory's FASTA.

    That happens when the CSV does not belong to the cluster run, or when its uids are
    spelled differently from the ones ``cluster`` wrote (task 0001.z9r: upper-case uids
    against lower-cased headers). Without this check every complex is skipped and the
    command reports empty clusters, folds or leaks with exit 0.
    """
    if complexes and not any(k in known_chains for c in complexes for k in c.chain_keys()):
        sample = ", ".join(sorted(known_chains)[:3])
        print(
            f"Error: {command}: none of the {len(complexes)} complexes has a chain in the "
            f"cluster directory's FASTA (it holds e.g. {sample}). Was the cluster run made "
            "from this CSV, with uids spelled the same way?",
            file=sys.stderr,
        )
        sys.exit(1)


def _cluster_chain_keys(cluster_dir: Path) -> set[str]:
    """Chain keys written by ``cluster``: the query FASTA, plus the target FASTA if any."""
    keys = read_fasta_keys(cluster_dir / FASTA_FILENAME)
    target = cluster_dir / TARGET_FASTA_FILENAME
    return keys | read_fasta_keys(target) if target.exists() else keys


# ---------------------------------------------------------------------------
# Subcommand: cluster
# ---------------------------------------------------------------------------


def _predicate_config_from_args(args: argparse.Namespace) -> dict:
    """Serialize the user's predicate selection to a dict for params.json."""
    if args.predicate == "rectangle":
        return {
            "kind": "rectangle",
            "min_seq_id": args.min_seq_id,
            "coverage": args.coverage,
            "cov_mode": args.cov_mode,
        }
    if args.predicate == "hyperbola":
        if args.hyperbola_x is None or not 0.0 < args.hyperbola_x <= 1.0:
            print(
                f"Error: --hyperbola-x must be in (0, 1], got {args.hyperbola_x!r} "
                "(canonical value: 0.24)",
                file=sys.stderr,
            )
            sys.exit(2)
        return {
            "kind": "hyperbola",
            "x": args.hyperbola_x,
            "cov_mode": args.cov_mode,
            "min_fident": getattr(args, "min_fident", 0.0),
        }
    raise ValueError(f"Unknown predicate kind: {args.predicate!r}")


def cmd_cluster(args: argparse.Namespace) -> None:
    csv_paths = [Path(p) for p in args.csv]
    target_csv_paths = [Path(p) for p in (args.target_csv or [])]
    pdb_dir = Path(args.pdb_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = args.sequence_source

    # Validate predicate flags eagerly — we don't want to extract sequences
    # and run mmseqs only to fail at the params.json write at the end.
    predicate_cfg = _predicate_config_from_args(args)

    print(f"Loading annotations from {len(csv_paths)} CSV(s)...")
    complexes = load_annotations(*csv_paths)
    print(f"  {len(complexes)} unique complexes")

    print(f"Extracting sequences from PDB files ({source})...")
    fasta_path = output_dir / FASTA_FILENAME
    n_seqs = write_fasta(complexes, pdb_dir, fasta_path, sequence_source=source)
    print(f"  Wrote {n_seqs} chain sequences to {fasta_path}")

    if target_csv_paths:
        # Purpose-split: query DB and target DB are different questions'
        # databases and are never merged into one universe.
        target_complexes = load_annotations(*target_csv_paths)
        target_fasta_path = output_dir / TARGET_FASTA_FILENAME
        n_target_seqs = write_fasta(
            target_complexes, pdb_dir, target_fasta_path, sequence_source=source
        )
        print(
            f"  Wrote {n_target_seqs} target chain sequences from "
            f"{len(target_complexes)} complexes to {target_fasta_path}"
        )
        print("Running mmseqs2 easy-search (query vs target, bidirectional)...")
        hits = run_mmseqs_search_pair(fasta_path, target_fasta_path, output_dir)
        scheme = {
            "kind": "purpose-split",
            "query_csvs": [str(p) for p in csv_paths],
            "target_csvs": [str(p) for p in target_csv_paths],
            "bidirectional": True,
            "n_target_sequences": n_target_seqs,
        }
        print(f"  {len(hits)} hits (both directions merged, query = query DB)")
    else:
        print("Running mmseqs2 easy-search (self-vs-self)...")
        hits = run_mmseqs_search_allvall(fasta_path, output_dir)
        scheme = {
            "kind": "self",
            "query_csvs": [str(p) for p in csv_paths],
            "target_csvs": [str(p) for p in csv_paths],
            "bidirectional": True,
        }
        print(f"  {len(hits)} hits (after self-removal and symmetrization)")

    hits_path = output_dir / HITS_FILENAME
    hits.to_parquet(hits_path)
    print(f"  Wrote hit table to {hits_path}")

    params = {
        "post_filter": predicate_cfg,
        "search_settings": SEARCH_SETTINGS,
        "sequence_source": source,
        "scheme": scheme,
        "n_sequences": n_seqs,
        "n_hits": len(hits),
        "csv_files": [str(p) for p in csv_paths],
        "pdb_dir": str(pdb_dir),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / PARAMS_FILENAME).write_text(json.dumps(params, indent=2))
    print(f"\nDone. Outputs in {output_dir}/")


# ---------------------------------------------------------------------------
# Subcommand: split
# ---------------------------------------------------------------------------


def _load_predicate_from_params(params: dict) -> EdgePredicate:
    pf = params["post_filter"]
    kind = pf.get("kind", "rectangle")
    if kind == "rectangle":
        return RectanglePredicate(
            min_pident=float(pf["min_seq_id"]),
            min_coverage=float(pf["coverage"]),
            cov_mode=int(pf["cov_mode"]),
        )
    if kind == "hyperbola":
        return HyperbolaPredicate(
            x=float(pf["x"]),
            cov_mode=int(pf["cov_mode"]),
            min_fident=float(pf.get("min_fident", 0.0)),
        )
    raise ValueError(f"Unknown predicate kind in params.json: {kind!r}")


def cmd_split(args: argparse.Namespace) -> None:
    cluster_dir = Path(args.cluster_dir)
    output_path = Path(args.output)

    params = json.loads((cluster_dir / PARAMS_FILENAME).read_text())
    predicate = _load_predicate_from_params(params)
    print(f"Post-filter: {predicate}")

    hits = pd.read_parquet(cluster_dir / HITS_FILENAME)
    edges = hits[predicate(hits)]
    print(f"  {len(edges)} similarity edges out of {len(hits)} raw hits")

    all_chains = read_fasta_keys(cluster_dir / FASTA_FILENAME)

    train_complexes = load_annotations(Path(args.csv))
    test_csv_paths = [Path(p) for p in (args.test_csv or [])]
    test_complexes = load_annotations(*test_csv_paths) if test_csv_paths else []
    all_complexes = train_complexes + test_complexes
    require_known_chains(all_complexes, all_chains, "split")

    # Complex adjacency comes from the direct chain edges. Chain clusters are
    # NOT an input here — cluster co-membership is a transitive closure and is
    # not a similarity relation. See ppisplit.graph.
    uid_to_cluster = complex_clusters_from_edges(
        all_complexes, edges,
        similarity_mode=args.similarity_mode,
        known_chains=all_chains,
    )
    grouped = group_clusters(uid_to_cluster)
    print(f"  {len(grouped)} complex-level clusters from {len(uid_to_cluster)} complexes")

    if test_complexes and args.exclude_test_collateral:
        test_uids = {c.uid for c in test_complexes}
        trainval, test, collateral = exclude_test_clusters(grouped, test_uids)
        print(
            f"  Excluding {len(test)} test-touched clusters from trainval "
            f"({sum(len(v) for v in test.values())} UIDs, "
            f"{len(collateral)} collateral train UIDs removed)"
        )
        grouped = trainval
    elif test_complexes:
        # Test sets passed for cluster-graph context, but train pool is kept whole.
        # Use 'clean-test-set' subcommand to produce a leak-free test CSV.
        test_uids = {c.uid for c in test_complexes}
        _, test, collateral = exclude_test_clusters(grouped, test_uids)
        print(
            f"  Note: {len(test)} test-touched clusters span "
            f"{len(collateral)} train UIDs that share sequence clusters with test. "
            "These are KEPT in trainval (use --exclude-test-collateral to drop them, "
            "or 'clean-test-set' to remove leaks from the test side)."
        )
        # Drop test UIDs from the trainval pool — they are not training data.
        grouped = {
            cid: [u for u in uids if u not in test_uids]
            for cid, uids in grouped.items()
        }
        grouped = {cid: uids for cid, uids in grouped.items() if uids}

    if args.weights and args.mode != "montecarlo-balanced":
        print("Error: --weights applies to montecarlo-balanced mode only", file=sys.stderr)
        sys.exit(1)

    if args.mode == "kfold":
        if not args.n_folds:
            print("Error: --n-folds required for kfold mode", file=sys.stderr)
            sys.exit(1)
        folds_df = assign_kfold(grouped, n_folds=args.n_folds, seed=args.seed)
    elif args.mode == "montecarlo":
        if not args.n_splits:
            print("Error: --n-splits required for montecarlo mode", file=sys.stderr)
            sys.exit(1)
        folds_df = assign_montecarlo(
            grouped, n_splits=args.n_splits, val_ratio=args.val_ratio, seed=args.seed
        )
    elif args.mode == "montecarlo-balanced":
        if not args.n_splits:
            print("Error: --n-splits required for montecarlo-balanced mode", file=sys.stderr)
            sys.exit(1)
        weights = None
        if args.weights:
            table = pd.read_csv(args.weights, dtype={"uid": str})
            weights = dict(zip(table["uid"], table["weight"].astype(float)))
        folds_df = assign_montecarlo_balanced(
            grouped, n_splits=args.n_splits, val_ratio=args.val_ratio, seed=args.seed,
            large_threshold=args.large_threshold, weights=weights,
        )
    else:
        print(f"Error: unknown mode {args.mode}", file=sys.stderr)
        sys.exit(1)

    folds_df.to_csv(output_path, index=False)
    print(f"  Wrote folds to {output_path}")


# ---------------------------------------------------------------------------
# Subcommand: check-leaks
# ---------------------------------------------------------------------------


def cmd_check_leaks(args: argparse.Namespace) -> None:
    cluster_dir = Path(args.cluster_dir)
    params = json.loads((cluster_dir / PARAMS_FILENAME).read_text())
    predicate = _load_predicate_from_params(params)
    print(f"Post-filter: {predicate}")

    hits = pd.read_parquet(cluster_dir / HITS_FILENAME)

    train_complexes = load_annotations(Path(args.train_csv))
    test_complexes = load_annotations(Path(args.test_csv))
    train_uids = {c.uid for c in train_complexes}
    test_uids = {c.uid for c in test_complexes}
    chain_role = chain_role_map([*train_complexes, *test_complexes])
    known = _cluster_chain_keys(cluster_dir)
    require_known_chains(test_complexes, known, "check-leaks")
    require_known_chains(train_complexes, known, "check-leaks")

    report = detect_leaks(
        hits=hits,
        predicate=predicate,
        chain_role=chain_role,
        test_uids=test_uids,
        train_uids=train_uids,
        similarity_mode=args.similarity_mode,
    )

    if report.leaking_pairs.empty:
        print(
            f"\nNo leaks found (mode={args.similarity_mode}) between "
            f"{len(test_uids)} test and {len(train_uids)} train complexes."
        )
        sys.exit(0)

    print(
        f"\nFound {len(report.leaking_train)} leaking train UIDs "
        f"with {len(report.leaking_test)} test UIDs (mode={args.similarity_mode}):\n"
    )

    complex_map = {c.uid: c for c in [*train_complexes, *test_complexes]}
    for _, pair in report.leaking_pairs.iterrows():
        t_ci = complex_map[pair["test_uid"]]
        n_ci = complex_map[pair["train_uid"]]
        print(
            f"  {pair['test_uid']} (R:{''.join(t_ci.receptor_chains)} "
            f"L:{''.join(t_ci.ligand_chains)}) <-> "
            f"{pair['train_uid']} (R:{''.join(n_ci.receptor_chains)} "
            f"L:{''.join(n_ci.ligand_chains)})"
        )
        pair_hits = report.evidence[
            (report.evidence["query_uid"] == pair["test_uid"])
            & (report.evidence["target_uid"] == pair["train_uid"])
        ]
        if pair["test_uid"] == pair["train_uid"]:
            print("      direct UID overlap — identical PDB in test and train")
        for _, hit in pair_hits.iterrows():
            if hit.get("leak_type") == "direct-uid":
                continue
            match_type = "same" if hit["query_role"] == hit["target_role"] else "cross"
            print(
                f"    {hit['query_role']:>8s}~{hit['target_role']:<8s} ({match_type}): "
                f"{hit['query']} ~ {hit['target']}  "
                f"identity={hit['fident'] * 100:.1f}%  alnlen={int(hit['alnlen'])}  "
                f"qcov={hit['qcov']:.2f}  tcov={hit['tcov']:.2f}"
            )

    if args.output:
        out_path = Path(args.output)
        out_cols = [
            "query_uid", "query_chain", "target_uid", "target_chain",
            "query_role", "target_role",
            "pident", "fident", "alnlen", "qlen", "tlen", "qcov", "tcov", "evalue",
            "leak_type",
        ]
        cols_present = [c for c in out_cols if c in report.evidence.columns]
        report.evidence.rename(
            columns={
                "query_uid": "test_uid", "target_uid": "train_uid",
                "query_chain": "test_chain", "target_chain": "train_chain",
            }
        )[
            [c.replace("query_uid", "test_uid").replace("target_uid", "train_uid")
               .replace("query_chain", "test_chain").replace("target_chain", "train_chain")
             for c in cols_present]
        ].to_csv(out_path, index=False)
        print(f"\nDetailed leak evidence written to {out_path}")

    if args.output_clean:
        train_df = pd.read_csv(args.train_csv)
        clean_df = train_df[~train_df["uid"].isin(report.leaking_train)]
        clean_path = Path(args.output_clean)
        clean_df.to_csv(clean_path, index=False)
        print(
            f"Clean train CSV ({len(clean_df)} rows, "
            f"{len(train_df) - len(clean_df)} removed) written to {clean_path}"
        )

    sys.exit(1)


# ---------------------------------------------------------------------------
# Subcommand: clean-test-set
# ---------------------------------------------------------------------------


def cmd_clean_test_set(args: argparse.Namespace) -> None:
    cluster_dir = Path(args.cluster_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params = json.loads((cluster_dir / PARAMS_FILENAME).read_text())
    predicate = _load_predicate_from_params(params)
    print(f"Post-filter: {predicate}")

    hits = pd.read_parquet(cluster_dir / HITS_FILENAME)
    test_complexes = load_annotations(Path(args.test_csv))
    test_uids = {c.uid for c in test_complexes}
    require_known_chains(test_complexes, _cluster_chain_keys(cluster_dir), "clean-test-set")
    test_df = pd.read_csv(args.test_csv)

    train_csv_paths = [Path(p) for p in args.train_csv]
    leaking_test_by_train: dict[str, set[str]] = {}

    for train_path in train_csv_paths:
        train_complexes = load_annotations(train_path)
        train_uids = {c.uid for c in train_complexes}
        chain_role = chain_role_map([*train_complexes, *test_complexes])
        require_known_chains(train_complexes, _cluster_chain_keys(cluster_dir), "clean-test-set")

        report = detect_leaks(
            hits=hits,
            predicate=predicate,
            chain_role=chain_role,
            test_uids=test_uids,
            train_uids=train_uids,
            similarity_mode=args.similarity_mode,
        )

        train_label = train_path.stem
        per_train_path = output_dir / f"leaks_vs_{train_label}.csv"
        if not report.evidence.empty:
            evidence = report.evidence.rename(columns={
                "query_uid": "test_uid", "target_uid": "train_uid",
                "query_chain": "test_chain", "target_chain": "train_chain",
            })
            keep_cols = [
                "test_uid", "test_chain", "train_uid", "train_chain",
                "query_role", "target_role",
                "fident", "pident", "alnlen", "qlen", "tlen", "qcov", "tcov", "evalue",
            ]
            evidence[[c for c in keep_cols if c in evidence.columns]].to_csv(per_train_path, index=False)
        else:
            pd.DataFrame().to_csv(per_train_path, index=False)

        leaking_test_by_train[train_label] = report.leaking_test
        print(f"  {train_label:30s} {len(report.leaking_test):>4d} test UIDs leak  "
              f"(evidence -> {per_train_path.name})")

    # Combined summary: test_uid -> set of train labels it leaked against
    union_leaking_test: set[str] = set()
    for s in leaking_test_by_train.values():
        union_leaking_test |= s

    summary_rows = []
    for uid in sorted(union_leaking_test):
        labels = sorted(name for name, s in leaking_test_by_train.items() if uid in s)
        summary_rows.append({"test_uid": uid, "leaking_train_sets": ",".join(labels)})
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "all_leaks_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Wide-format leaks.csv: one row per test UID, binary column per train set.
    # Drop-in replacement for the historical build_leaks_csv.py output.
    wide_rows = []
    for uid in test_df["uid"].astype(str).tolist():
        row = {"uid": uid}
        for train_label in sorted(leaking_test_by_train):
            row[f"{train_label}_leak"] = int(uid in leaking_test_by_train[train_label])
        wide_rows.append(row)
    wide_df = pd.DataFrame(wide_rows)
    wide_path = output_dir / "leaks.csv"
    wide_df.to_csv(wide_path, index=False)

    # Clean test CSV
    clean_test = test_df[~test_df["uid"].isin(union_leaking_test)]
    clean_test_path = output_dir / f"{Path(args.test_csv).stem}_clean.csv"
    clean_test.to_csv(clean_test_path, index=False)

    # Clean params for reproducibility
    clean_params = {
        "test_csv": str(args.test_csv),
        "train_csvs": [str(p) for p in train_csv_paths],
        "cluster_dir": str(cluster_dir),
        "post_filter": params["post_filter"],
        "similarity_mode": args.similarity_mode,
        "test_uids_total": len(test_df),
        "test_uids_leaking": len(union_leaking_test),
        "test_uids_clean": len(clean_test),
        "leaking_per_train": {k: len(v) for k, v in leaking_test_by_train.items()},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "clean_params.json").write_text(json.dumps(clean_params, indent=2))

    print(f"\nUnion of leaking test UIDs: {len(union_leaking_test)}")
    print(f"Clean test set: {len(clean_test)} / {len(test_df)} rows -> {clean_test_path}")
    print(f"Per-train summary: {summary_path}")


# ---------------------------------------------------------------------------
# Subcommand: summary
# ---------------------------------------------------------------------------


def cmd_summary(args: argparse.Namespace) -> None:
    cluster_dir = Path(args.cluster_dir)
    params = json.loads((cluster_dir / PARAMS_FILENAME).read_text())
    predicate = _load_predicate_from_params(params)
    hits = pd.read_parquet(cluster_dir / HITS_FILENAME)
    edges = hits[predicate(hits)]
    all_chains = read_fasta_keys(cluster_dir / FASTA_FILENAME)

    if args.chain or not args.csv:
        # Chain clusters are a DIAGNOSTIC here and nothing downstream consumes
        # them — they are the transitive closure of similarity, not similarity.
        chain_clusters = chain_clusters_from_edges(edges, all_chains)
        sizes = [len(v) for v in _group_chain_clusters(chain_clusters).values()]
        print(f"Chain-level clusters: {len(sizes)} clusters from {len(chain_clusters)} chains")
        _print_size_distribution(sizes)
        if not args.csv:
            return

    csv_paths = [Path(p) for p in (args.csv or [])]
    train_complexes = load_annotations(*csv_paths)
    test_complexes: list = []
    if args.test_csv:
        test_paths = [Path(p) for p in args.test_csv]
        test_complexes = load_annotations(*test_paths)

    all_complexes = train_complexes + test_complexes
    require_known_chains(all_complexes, all_chains, "summary")
    uid_to_cluster = complex_clusters_from_edges(
        all_complexes, edges,
        similarity_mode=args.similarity_mode,
        known_chains=all_chains,
    )
    grouped = group_clusters(uid_to_cluster)

    if test_complexes:
        test_uids = {c.uid for c in test_complexes}
        trainval, test_clusters, collateral = exclude_test_clusters(grouped, test_uids)
        print(f"\nComplex clusters (mode={args.similarity_mode}):")
        print(f"  Total: {len(grouped)} clusters, {len(uid_to_cluster)} UIDs")
        print(f"  Trainval: {len(trainval)} clusters, "
              f"{sum(len(v) for v in trainval.values())} UIDs")
        print(f"  Test (with collateral): {len(test_clusters)} clusters, "
              f"{sum(len(v) for v in test_clusters.values())} UIDs "
              f"({len(collateral)} train collateral)")
    else:
        trainval = grouped

    sizes = [len(v) for v in trainval.values()]
    print("\nTrainval cluster size distribution:")
    _print_size_distribution(sizes)

    if args.top:
        print(f"\nTop {args.top} largest clusters:")
        for cid, uids in sorted(trainval.items(), key=lambda kv: -len(kv[1]))[: args.top]:
            print(f"  cluster {cid} ({len(uids)} UIDs): {uids[:8]}{' ...' if len(uids) > 8 else ''}")


def _group_chain_clusters(chain_clusters: dict[str, int]) -> dict[int, list[str]]:
    grouped: dict[int, list[str]] = defaultdict(list)
    for ch, cid in chain_clusters.items():
        grouped[cid].append(ch)
    return dict(grouped)


def _print_size_distribution(sizes: list[int]) -> None:
    bins: list[tuple[int, int | None]] = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 50), (51, 100), (101, None)]
    print("  Size distribution:")
    for lo, hi in bins:
        if hi is None:
            n = sum(1 for s in sizes if s >= lo)
            label = f"{lo}+"
        else:
            n = sum(1 for s in sizes if lo <= s <= hi)
            label = f"{lo}-{hi}" if lo != hi else f"{lo}"
        print(f"    {label:>6s}: {n}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ppisplit",
        description="Sequence-aware CV splitting and leak detection for PPI datasets.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # cluster
    p_c = sub.add_parser("cluster", help="Run mmseqs2 search and cache the hit table")
    p_c.add_argument("--csv", required=True, action="append",
                     help="Query-side annotation CSV (repeatable). Without --target-csv "
                          "this is a self-vs-self search (fold clustering, redundancy).")
    p_c.add_argument("--target-csv", action="append", default=[],
                     help="Target-side annotation CSV (repeatable). Given, the search "
                          "becomes purpose-split: query = --csv set(s), target = these, "
                          "run in both directions. One (query, target) pair per question; "
                          "never pool, so adding a set cannot move another's verdicts.")
    p_c.add_argument("--pdb-dir", required=True, help="Directory with <uid>.pdb, .cif, .pdb.gz or .cif.gz files")
    p_c.add_argument("--output-dir", required=True, help="Output directory")
    p_c.add_argument("--sequence-source", default=DEFAULT_SEQUENCE_SOURCE,
                     choices=list(SEQUENCE_SOURCES),
                     help="Which sequence to compare: 'seqres-trimmed' (default, SEQRES from the "
                          "first to the last resolved residue), 'resolved' (residues "
                          "present in the model) or 'seqres' (full entity sequence, "
                          "including unresolved tags and termini). 'resolved' and "
                          "'seqres' reproduce older verdicts.")
    p_c.add_argument("--predicate", default="hyperbola", choices=["hyperbola", "rectangle"],
                     help="Post-filter predicate kind (default: hyperbola, the canonical rule; "
                          "rectangle is legacy, for reproducing historical numbers)")
    p_c.add_argument("--min-seq-id", type=float, default=0.3,
                     help="Rectangle (legacy): sequence-identity threshold (default: 0.3)")
    p_c.add_argument("--coverage", type=float, default=0.8,
                     help="Rectangle (legacy): coverage threshold (default: 0.8)")
    p_c.add_argument("--cov-mode", type=int, default=COV_MODE_SYMMETRIC,
                     help="Coverage mode: -1=max(qcov,tcov) (orientation invariant, canonical), "
                          "0=both, 1=tcov, 2=qcov (default: -1)")
    p_c.add_argument("--hyperbola-x", type=float, default=0.24,
                     help="Hyperbola: fident*cov >= x (default: 0.24, the canonical threshold)")
    p_c.add_argument("--min-fident", type=float, default=0.0,
                     help="Hyperbola: additional identity floor on fident. Only for "
                          "test-internal redundancy screening; leave 0 for leak detection.")
    p_c.set_defaults(func=cmd_cluster)

    # split
    p_s = sub.add_parser("split", help="Build complex clusters and assign folds")
    p_s.add_argument("--csv", required=True, help="Train/val annotation CSV")
    p_s.add_argument("--cluster-dir", required=True, help="Directory from cluster step")
    p_s.add_argument("--output", required=True, help="Output folds CSV")
    p_s.add_argument("--mode", required=True, choices=["kfold", "montecarlo", "montecarlo-balanced"])
    p_s.add_argument("--n-folds", type=int)
    p_s.add_argument("--n-splits", type=int)
    p_s.add_argument("--val-ratio", type=float, default=0.25)
    p_s.add_argument("--seed", type=int, default=42)
    p_s.add_argument("--large-threshold", type=float, default=15,
                     help="montecarlo-balanced: a cluster larger than this (in weight units) "
                          "goes to val by a coin flip, the smaller ones fill val up to "
                          "--val-ratio (default: 15)")
    p_s.add_argument("--weights",
                     help="montecarlo-balanced: CSV with columns uid, weight; a cluster's size "
                          "is the sum of its uids' weights (default: 1 per uid)")
    p_s.add_argument("--similarity-mode", default="both",
                     choices=["both", "receptor-only", "ligand-only", "either"])
    p_s.add_argument("--test-csv", action="append", default=[],
                     help="Test CSV(s) (repeatable). Test UIDs are dropped from the train pool, "
                          "but train UIDs that share sequence clusters with test are KEPT by default.")
    p_s.add_argument("--exclude-test-collateral", action="store_true",
                     help="Also drop train UIDs caught in test-touched clusters. "
                          "Default: false. Prefer running 'clean-test-set' to remove leaks "
                          "from the test side; collateral exclusion is the dual operation "
                          "(remove from train) and is generally redundant with a cleaned test set.")
    p_s.set_defaults(func=cmd_split)

    # check-leaks
    p_l = sub.add_parser("check-leaks", help="Detect (test_uid, train_uid) leaks")
    p_l.add_argument("--train-csv", required=True)
    p_l.add_argument("--test-csv", required=True)
    p_l.add_argument("--cluster-dir", required=True,
                     help="Directory from cluster step. For a final verdict use the "
                          "purpose-split search for this question: "
                          "'cluster --csv <test> --target-csv <train>'.")
    p_l.add_argument("--similarity-mode", default="both",
                     choices=["both", "receptor-only", "ligand-only", "either"],
                     help="'both' (default) is the canonical rule: a leak requires BOTH partners "
                          "to match the same train complex. The other modes are diagnostics.")
    p_l.add_argument("--output", help="Output detailed leaks CSV")
    p_l.add_argument("--output-clean", help="Output filtered train CSV (leaks removed)")
    p_l.set_defaults(func=cmd_check_leaks)

    # clean-test-set
    p_ct = sub.add_parser("clean-test-set", help="Produce a clean test CSV against multiple train sets")
    p_ct.add_argument("--test-csv", required=True)
    p_ct.add_argument("--train-csv", required=True, action="append",
                      help="Train annotation CSV (repeatable; test UIDs leaking against any train set are removed)")
    p_ct.add_argument("--cluster-dir", required=True,
                      help="Directory from cluster step. Use the purpose-split search for "
                           "this question — see check-leaks.")
    p_ct.add_argument("--output-dir", required=True)
    p_ct.add_argument("--similarity-mode", default="both",
                      choices=["both", "receptor-only", "ligand-only", "either"],
                      help="'both' (default) is the canonical rule.")
    p_ct.set_defaults(func=cmd_clean_test_set)

    # summary
    p_sum = sub.add_parser("summary", help="Print cluster size statistics")
    p_sum.add_argument("--cluster-dir", required=True)
    p_sum.add_argument("--csv", action="append")
    p_sum.add_argument("--test-csv", action="append")
    p_sum.add_argument("--chain", action="store_true", help="Include chain-level stats")
    p_sum.add_argument("--similarity-mode", default="both",
                       choices=["both", "receptor-only", "ligand-only", "either"])
    p_sum.add_argument("--top", type=int, default=0,
                       help="Print N largest trainval clusters")
    p_sum.set_defaults(func=cmd_summary)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
