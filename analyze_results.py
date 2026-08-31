from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = ["objective_return", "fish_eaten", "success_rate", "death_rate"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/runs.csv")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    with open(args.input, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("No completed runs found")
    groups = defaultdict(list)
    for row in rows:
        key = row['method'] if 'method' in row else (row["reward"], row["exploration"], row["architecture"])
        groups[key].append(row)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for key, values in sorted(groups.items()):
        if len({v['seed'] for v in values}) != len(values):
            raise ValueError('Duplicate seeds in one method; refusing pseudoreplication')
        summary = ({'method': key} if isinstance(key, str) else
                   {"reward": key[0], "exploration": key[1], "architecture": key[2]})
        summary['n_seeds'] = len(values)
        for metric in METRICS:
            x = np.asarray([float(v[metric]) for v in values])
            summary[f"{metric}_mean"] = float(x.mean())
            summary[f"{metric}_std"] = float(x.std(ddof=1)) if len(x) > 1 else float("nan")
        summary_rows.append(summary)
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader(); writer.writerows(summary_rows)

    labels = [r['method'] if 'method' in r else f"{r['reward']}\n{r['exploration']}\n{r['architecture']}" for r in summary_rows]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, metric in zip(axes.flat, METRICS):
        means = [r[f"{metric}_mean"] for r in summary_rows]
        errors = [r[f"{metric}_std"] for r in summary_rows]
        if any(np.isfinite(errors)):
            ax.bar(range(len(labels)), means, yerr=errors, capsize=4, color="#377eb8")
        else:
            ax.bar(range(len(labels)), means, color="#377eb8")
        ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
        ax.set_title(f"{metric} (mean +/- sample SD; absent for n=1)")
        ax.grid(axis="y", alpha=.25)
    fig.suptitle("FishRL: held-out task evaluation (mean +/- SD across seeds)")
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=180)
    print(f"Wrote {out / 'summary.csv'} and {out / 'comparison.png'}")


if __name__ == "__main__":
    main()
