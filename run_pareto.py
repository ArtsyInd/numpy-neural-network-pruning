"""Part 4: repeated sparsity/accuracy/cost experiments for the NumPy MLP.

Default sweep: 0%, 50%, 75%, 90%, 95% target sparsity; five seeds;
Taylor and magnitude pruning at non-zero targets plus an unpruned baseline.
All training, gradients, optimization, pruning, and sparse inference use the
project's own NumPy implementation.
"""
import argparse
import csv
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tensor import Tensor
from layers import Linear
from masked_parameter import MaskedAdam
from optim import Adam
from pruning import GlobalPruner, cubic_sparsity
from sparse_inference import predict_sparse, dense_zero_predict
from train_part2 import make_spiral_data, evaluate
from train_part3 import make_prunable_model


def prepare_data():
    """Fixed data and split make method comparisons paired and reproducible."""
    X, y = make_spiral_data(seed=42)
    split_rng = np.random.default_rng(99)
    indices = split_rng.permutation(len(y))
    split = int(0.8 * len(y))
    train_idx, test_idx = indices[:split], indices[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    std[std == 0] = 1.0
    return ((X_train - mean) / std, y_train,
            (X_test - mean) / std, y_test)


def train_one(target_sparsity, method, seed, X_train, y_train,
              X_test, y_test, epochs=300, batch_size=64,
              learning_rate=0.003, prune_start_fraction=0.20,
              prune_end_fraction=0.80, prune_every=5):
    model = make_prunable_model(seed)
    masked_params = model.masked_parameters()
    pruner = GlobalPruner(masked_params, method=method) if method != "none" else None
    masked_opt = MaskedAdam(masked_params, learning_rate=learning_rate)
    dense_opt = Adam(model.dense_parameters(), learning_rate=learning_rate)
    start = int(epochs * prune_start_fraction)
    end = max(start + 1, int(epochs * prune_end_fraction))
    batch_rng = np.random.default_rng(seed + 1000)

    for epoch in range(1, epochs + 1):
        order = batch_rng.permutation(len(y_train))
        for pos in range(0, len(order), batch_size):
            ids = order[pos:pos + batch_size]
            masked_opt.zero_grad()
            dense_opt.zero_grad()
            loss = model(Tensor(X_train[ids])).softmax_cross_entropy(y_train[ids])
            loss_value = float(np.asarray(loss.data))
            if not np.isfinite(loss_value):
                raise FloatingPointError(f"Non-finite loss: seed={seed}, epoch={epoch}")
            loss.backward()

            if (pruner is not None and target_sparsity > 0 and start < end
                    and start <= epoch <= end
                    and (epoch - start) % prune_every == 0):
                desired = cubic_sparsity(epoch, start, end, target_sparsity)
                pruner.prune_to_sparsity(desired)

            masked_opt.step()
            dense_opt.step()

    train_loss, train_accuracy = evaluate(model, X_train, y_train)
    test_loss, test_accuracy = evaluate(model, X_test, y_test)
    stats = (pruner.statistics() if pruner is not None else {
        "total_connections": sum(p.mask.size for p in masked_params),
        "active_connections": sum(int(p.mask.sum()) for p in masked_params),
        "pruned_connections": 0,
        "sparsity": 0.0,
    })
    return model, {
        "seed": seed,
        "method": method,
        "target_sparsity": float(target_sparsity),
        "achieved_sparsity": float(stats["sparsity"]),
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "active_weight_connections": int(stats["active_connections"]),
        "total_weight_connections": int(stats["total_connections"]),
    }


def median_latency_ms(function, warmups=2, repeats=7):
    for _ in range(warmups):
        function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000.0)
    return float(statistics.median(samples))


def measure_cost(model, X_test, timing_repeats=7):
    sparse_output, active_multiplies = predict_sparse(model, X_test)
    dense_output = dense_zero_predict(model, X_test)
    max_abs_error = float(np.max(np.abs(sparse_output - dense_output)))
    if not np.allclose(sparse_output, dense_output, rtol=1e-9, atol=1e-10):
        raise AssertionError(f"Sparse/dense inference mismatch: max abs error={max_abs_error}")

    batch_size = len(X_test)
    total_connections = sum(layer.weight.data.size for layer in model.layers
                            if isinstance(layer, Linear))
    dense_multiplies = int(batch_size * total_connections)
    active_connections = sum(int(layer.weight.mask.sum()) for layer in model.layers
                              if isinstance(layer, Linear) and layer.prunable)
    total_trainable = sum(p.data.size for p in model.parameters())
    active_trainable = active_connections + sum(
        layer.bias.data.size for layer in model.layers if isinstance(layer, Linear)
    )
    sparse_ms = median_latency_ms(lambda: predict_sparse(model, X_test), repeats=timing_repeats)
    dense_ms = median_latency_ms(lambda: dense_zero_predict(model, X_test), repeats=timing_repeats)
    return {
        "active_weight_connections": int(active_connections),
        "total_weight_connections": int(total_connections),
        "active_trainable_parameters": int(active_trainable),
        "total_trainable_parameters": int(total_trainable),
        "test_batch_active_multiply_count": int(active_multiplies),
        "test_batch_dense_multiply_count": int(dense_multiplies),
        "multiply_count_reduction_fraction": (1.0 - active_multiplies / dense_multiplies)
            if dense_multiplies else 0.0,
        "sparse_inference_median_ms": sparse_ms,
        "dense_zero_filled_inference_median_ms": dense_ms,
        "dense_over_sparse_latency_ratio": dense_ms / sparse_ms if sparse_ms else math.nan,
        "sparse_dense_max_abs_error": max_abs_error,
    }


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(raw_rows):
    groups = defaultdict(list)
    for row in raw_rows:
        groups[(float(row["target_sparsity"]), row["method"])].append(row)
    out = []
    numeric_fields = ["achieved_sparsity", "test_accuracy", "test_loss",
                      "active_weight_connections", "active_trainable_parameters",
                      "multiply_count_reduction_fraction", "sparse_inference_median_ms",
                      "dense_zero_filled_inference_median_ms", "dense_over_sparse_latency_ratio"]
    for (target, method), rows in sorted(groups.items()):
        entry = {"target_sparsity": target, "method": method, "n_seeds": len(rows)}
        for field in numeric_fields:
            vals = [float(r[field]) for r in rows]
            entry[field + "_mean"] = statistics.mean(vals)
            entry[field + "_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out.append(entry)
    return out


def paired_comparison(raw_rows, target):
    by_key = {(float(r["target_sparsity"]), r["method"], int(r["seed"])): r
              for r in raw_rows}
    seeds = sorted({int(r["seed"]) for r in raw_rows
                    if float(r["target_sparsity"]) == float(target)
                    and r["method"] in {"taylor", "magnitude"}})
    diffs = []
    for seed in seeds:
        t = by_key.get((float(target), "taylor", seed))
        m = by_key.get((float(target), "magnitude", seed))
        if t is not None and m is not None:
            diffs.append((float(t["test_accuracy"]) - float(m["test_accuracy"])) * 100.0)
    if not diffs:
        return None
    mean_diff = statistics.mean(diffs)
    sd_diff = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    half_width = 1.96 * sd_diff / math.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
    return {"n": len(diffs), "mean_diff_pp": mean_diff, "sd_diff_pp": sd_diff,
            "approx_ci_low_pp": mean_diff - half_width,
            "approx_ci_high_pp": mean_diff + half_width, "paired_differences_pp": diffs}


def write_report(path, raw_rows, summary_rows, target_for_claim=0.90):
    lines = [
        "# Part 4 — Pruning quality and inference cost",
        "",
        "## Experimental setup",
        "",
        "- Dataset: synthetic three-class spiral dataset generated with NumPy.",
        "- Train/test split and standardization are fixed across methods; scaling is fit on training data only.",
        "- Model: 2 → 64 → 64 → 3 MLP with ReLU hidden activations.",
        "- Optimizer: custom NumPy Adam / masked Adam; pruning schedule is cubic from 20% to 80% of training.",
        "- Importance methods: Taylor `abs(weight * gradient)` and magnitude `abs(weight)`.",
        "- Baseline: no pruning at 0% target sparsity.",
        "- Sparse inference skips inactive input connections by output neuron. Dense timing uses dense matrix multiplication over weight arrays whose pruned entries are exact zeros.",
        "- Latency is the median of repeated runs on the full test set. Results are hardware- and load-dependent; no speedup is assumed.",
        "",
        "## Aggregated results",
        "",
        "Accuracy is reported as a fraction; standard deviations are across seeds.",
        "",
        "| Target | Method | Seeds | Achieved sparsity (mean ± SD) | Test accuracy (mean ± SD) | Multiply reduction | Sparse ms | Dense zero-filled ms |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['target_sparsity']:.0%} | {r['method']} | {r['n_seeds']} | "
            f"{r['achieved_sparsity_mean']:.4f} ± {r['achieved_sparsity_std']:.4f} | "
            f"{r['test_accuracy_mean']:.4f} ± {r['test_accuracy_std']:.4f} | "
            f"{r['multiply_count_reduction_fraction_mean']:.1%} | "
            f"{r['sparse_inference_median_ms_mean']:.3f} | "
            f"{r['dense_zero_filled_inference_median_ms_mean']:.3f} |"
        )
    comp = paired_comparison(raw_rows, target_for_claim)
    lines.extend(["", f"## Quantitative claim at {target_for_claim:.0%} target sparsity", ""])
    if comp:
        trows = [r for r in raw_rows if float(r["target_sparsity"]) == target_for_claim and r["method"] == "taylor"]
        mrows = [r for r in raw_rows if float(r["target_sparsity"]) == target_for_claim and r["method"] == "magnitude"]
        tmean = statistics.mean(float(r["test_accuracy"]) for r in trows) * 100
        mmean = statistics.mean(float(r["test_accuracy"]) for r in mrows) * 100
        tsd = statistics.stdev(float(r["test_accuracy"]) for r in trows) * 100 if len(trows) > 1 else 0
        msd = statistics.stdev(float(r["test_accuracy"]) for r in mrows) * 100 if len(mrows) > 1 else 0
        lines.append(
            f"Across {comp['n']} paired seeds at {target_for_claim:.0%} target sparsity, "
            f"Taylor test accuracy was {tmean:.2f}% ± {tsd:.2f} percentage points, "
            f"versus {mmean:.2f}% ± {msd:.2f} for magnitude pruning. The paired mean difference "
            f"(Taylor − magnitude) was {comp['mean_diff_pp']:.2f} percentage points "
            f"(SD {comp['sd_diff_pp']:.2f}; approximate 95% normal interval "
            f"[{comp['approx_ci_low_pp']:.2f}, {comp['approx_ci_high_pp']:.2f}])."
        )
        if comp["approx_ci_low_pp"] <= 0 <= comp["approx_ci_high_pp"]:
            lines.append("\nThe approximate interval includes zero, so this sweep does not provide clear evidence that either method is better at this target. Treat the comparison as inconclusive, especially with a small number of seeds.")
        else:
            lines.append("\nThe approximate interval excludes zero in this experiment, which is preliminary evidence of a difference on this dataset. With only a small number of seeds and one synthetic task, it is not proof of general superiority.")
    else:
        lines.append(f"No paired Taylor/magnitude results were available at {target_for_claim:.0%}; rerun with that target included to generate the claim.")
    lines.extend([
        "",
        "## Interpretation limits",
        "",
        "- Multiply-count reduction is a structural work estimate: it counts only active weight multiplications on the sparse-aware path. It does not include all memory, indexing, bias, activation, or Python-loop costs.",
        "- The sparse-aware implementation may be slower in wall-clock time on small networks despite doing fewer multiply operations. Report the measured timings as observed, not as a guaranteed deployment speedup.",
        "- The approximate interval uses a normal 1.96 multiplier and is descriptive for this small repeated-seed experiment, not a definitive significance test.",
        "- Do not claim that the synthetic spiral result generalizes to real-world datasets without additional experiments.",
        "",
        "## Files",
        "",
        "- `part4_results_raw.csv`: one row per seed/method/target.",
        "- `part4_results_summary.csv`: mean and standard deviation by target and method.",
        "- `part4_pareto.png`: final test accuracy versus achieved sparsity with seed standard-deviation error bars.",
        "- `part4_report.md`: setup, results, cost measurements, and generated claim.",
    ])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_pareto(path, summary_rows):
    plt.figure(figsize=(8.5, 5.5))
    styles = {"none": "o-", "taylor": "s-", "magnitude": "^-"}
    labels = {"none": "Unpruned baseline", "taylor": "Taylor pruning", "magnitude": "Magnitude pruning"}
    for method in ("none", "taylor", "magnitude"):
        rows = sorted([r for r in summary_rows if r["method"] == method],
                      key=lambda r: r["achieved_sparsity_mean"])
        if not rows:
            continue
        xs = [r["achieved_sparsity_mean"] * 100 for r in rows]
        ys = [r["test_accuracy_mean"] * 100 for r in rows]
        errs = [r["test_accuracy_std"] * 100 for r in rows]
        plt.errorbar(xs, ys, yerr=errs, fmt=styles[method], capsize=3, label=labels[method])
    plt.xlabel("Achieved weight sparsity (%)")
    plt.ylabel("Final test accuracy (%)")
    plt.title("Part 4: sparsity–accuracy Pareto comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=0.003)
    p.add_argument("--targets", default="0,0.5,0.75,0.9,0.95",
                   help="Comma-separated target sparsities in [0,1), e.g. 0,0.5,0.75,0.9,0.95")
    p.add_argument("--seeds", default="11,22,33,44,55",
                   help="Comma-separated integer seeds")
    p.add_argument("--prune-start-fraction", type=float, default=0.20)
    p.add_argument("--prune-end-fraction", type=float, default=0.80)
    p.add_argument("--prune-every", type=int, default=5)
    p.add_argument("--timing-repeats", type=int, default=7)
    p.add_argument("--output-dir", default="part4_results")
    p.add_argument("--claim-target", type=float, default=0.90)
    return p.parse_args()


def main():
    args = parse_args()
    targets = sorted(set(float(x.strip()) for x in args.targets.split(",")))
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not targets or any(not 0 <= x < 1 for x in targets):
        raise ValueError("Targets must be a non-empty list of sparsities in [0,1)")
    if not seeds:
        raise ValueError("Provide at least one seed")
    if args.epochs < 2 or args.batch_size < 1 or args.prune_every < 1 or args.timing_repeats < 1:
        raise ValueError("epochs >= 2; batch size, prune interval, and timing repeats must be positive")
    if not 0 <= args.prune_start_fraction < args.prune_end_fraction <= 1:
        raise ValueError("Require 0 <= prune-start-fraction < prune-end-fraction <= 1")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    X_train, y_train, X_test, y_test = prepare_data()
    raw_rows = []
    planned = len(seeds) * (1 + 2 * sum(t > 0 for t in targets))
    run_num = 0

    for target in targets:
        methods = ["none"] if target == 0 else ["taylor", "magnitude"]
        for method in methods:
            for seed in seeds:
                run_num += 1
                print(f"[{run_num}/{planned}] target={target:.0%} method={method} seed={seed}", flush=True)
                model, row = train_one(
                    target, method, seed, X_train, y_train, X_test, y_test,
                    epochs=args.epochs, batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    prune_start_fraction=args.prune_start_fraction,
                    prune_end_fraction=args.prune_end_fraction,
                    prune_every=args.prune_every,
                )
                cost = measure_cost(model, X_test, timing_repeats=args.timing_repeats)
                row.update(cost)
                raw_rows.append(row)
                # Checkpoint raw results after each run so progress is not lost if interrupted.
                write_csv(outdir / "part4_results_raw.csv", raw_rows)
                print(
                    f"  test_acc={row['test_accuracy']:.4f}; sparsity={row['achieved_sparsity']:.4f}; "
                    f"active multiplies={row['test_batch_active_multiply_count']:,}/"
                    f"{row['test_batch_dense_multiply_count']:,}; sparse/dense ms="
                    f"{row['sparse_inference_median_ms']:.3f}/{row['dense_zero_filled_inference_median_ms']:.3f}",
                    flush=True,
                )

    summary_rows = aggregate(raw_rows)
    write_csv(outdir / "part4_results_summary.csv", summary_rows)
    plot_pareto(outdir / "part4_pareto.png", summary_rows)
    write_report(outdir / "part4_report.md", raw_rows, summary_rows, args.claim_target)
    print("\nPart 4 complete. Outputs:")
    for name in ("part4_results_raw.csv", "part4_results_summary.csv", "part4_pareto.png", "part4_report.md"):
        print(f"  {outdir / name}")


if __name__ == "__main__":
    main()
