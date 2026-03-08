#!/usr/bin/env python3
import argparse
import json
import os
from collections import defaultdict

from datasets import load_dataset, get_dataset_config_names
from tqdm import tqdm


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def open_jsonl(path: str, mode: str):
    ensure_dir(os.path.dirname(path))
    return open(path, mode, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="edinburgh-dawg/mmlu-redux-2.0")
    ap.add_argument("--out_dir", type=str, default="./mmlu_redux_2_jsonl")
    ap.add_argument("--streaming", action="store_true", help="use streaming mode (recommended)")
    ap.add_argument("--no_streaming", dest="streaming", action="store_false")
    ap.set_defaults(streaming=True)

    ap.add_argument(
        "--splits",
        type=str,
        default="",
        help="comma-separated splits to export, e.g. train,validation,test (empty=all)",
    )
    ap.add_argument(
        "--max_per_subset_per_split",
        type=int,
        default=0,
        help="limit per (subset,split); 0 means no limit (useful for quick test)",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing jsonl files in out_dir",
    )
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    wanted_splits = None
    if args.splits.strip():
        wanted_splits = set([s.strip() for s in args.splits.split(",") if s.strip()])

    # 1) enumerate subsets (configs)
    configs = get_dataset_config_names(args.dataset)
    if not configs:
        # Some datasets have no configs exposed; fallback to a single default config
        configs = [None]

    # 2) prepare output files (one per split, merged across subsets)
    out_files = {}  # split -> file handle
    out_paths = {}  # split -> path

    def get_fh(split: str):
        if split not in out_files:
            out_path = os.path.join(args.out_dir, f"{split}.jsonl")
            if args.overwrite and os.path.exists(out_path):
                os.remove(out_path)
            # append mode so multiple subsets can write into same split file
            out_files[split] = open_jsonl(out_path, "a")
            out_paths[split] = out_path
        return out_files[split]

    # 3) write records
    counts = defaultdict(lambda: defaultdict(int))  # counts[split][subset] = n
    total_counts = defaultdict(int)  # total per split

    print(f"[info] dataset={args.dataset}")
    print(f"[info] configs(subsets)={len(configs)}")
    print(f"[info] out_dir={args.out_dir}")
    print(f"[info] streaming={args.streaming}")
    if wanted_splits:
        print(f"[info] wanted_splits={sorted(list(wanted_splits))}")
    if args.max_per_subset_per_split > 0:
        print(f"[info] max_per_subset_per_split={args.max_per_subset_per_split}")

    for cfg in configs:
        subset_name = cfg if cfg is not None else "default"
        print(f"\n[subset] {subset_name}")

        # load as dict of splits
        ds = load_dataset(args.dataset, cfg, streaming=args.streaming)

        # ds is a DatasetDict / IterableDatasetDict-like
        available_splits = list(ds.keys())
        if wanted_splits:
            splits = [s for s in available_splits if s in wanted_splits]
        else:
            splits = available_splits

        if not splits:
            print(f"  [warn] no requested splits found in {available_splits}")
            continue

        for split in splits:
            fh = get_fh(split)

            # Iterate records
            limit = args.max_per_subset_per_split
            n = 0

            # tqdm for iterable datasets (streaming) can be tricky if no length; keep it simple
            iterator = ds[split]
            for ex in iterator:
                # Make it pure JSON-serializable dict (streaming gives python dict already)
                rec = dict(ex)

                # Add subset -> "type" (as you requested)
                # If the original example already has a "type", we preserve it as "orig_type"
                if "type" in rec and rec["type"] != subset_name:
                    rec["orig_type"] = rec["type"]
                rec["type"] = subset_name

                # Optional: keep dataset id for provenance
                rec["_hf_dataset"] = args.dataset
                rec["_hf_subset"] = subset_name
                rec["_hf_split"] = split

                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

                n += 1
                counts[split][subset_name] += 1
                total_counts[split] += 1

                if limit > 0 and n >= limit:
                    break

            print(f"  [split] {split}: wrote {n}")

    # close files
    for fh in out_files.values():
        fh.close()

    # 4) manifest
    manifest = {
        "dataset": args.dataset,
        "streaming": args.streaming,
        "out_files": out_paths,
        "totals_by_split": dict(total_counts),
        "counts_by_split_and_subset": {sp: dict(sub) for sp, sub in counts.items()},
    }
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n[done]")
    for sp in sorted(total_counts.keys()):
        print(f"  {sp}: total={total_counts[sp]} -> {out_paths.get(sp)}")
    print(f"  manifest: {manifest_path}")


if __name__ == "__main__":
    main()
