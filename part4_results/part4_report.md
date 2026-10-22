# Part 4 — Pruning quality and inference cost

## Experimental setup

- Dataset: synthetic three-class spiral dataset generated with NumPy.
- Train/test split and standardization are fixed across methods; scaling is fit on training data only.
- Model: 2 → 64 → 64 → 3 MLP with ReLU hidden activations.
- Optimizer: custom NumPy Adam / masked Adam; pruning schedule is cubic from 20% to 80% of training.
- Importance methods: Taylor `abs(weight * gradient)` and magnitude `abs(weight)`.
- Baseline: no pruning at 0% target sparsity.
- Sparse inference skips inactive input connections by output neuron. Dense timing uses dense matrix multiplication over weight arrays whose pruned entries are exact zeros.
- Latency is the median of repeated runs on the full test set. Results are hardware- and load-dependent; no speedup is assumed.

## Aggregated results

Accuracy is reported as a fraction; standard deviations are across seeds.

| Target | Method | Seeds | Achieved sparsity (mean ± SD) | Test accuracy (mean ± SD) | Multiply reduction | Sparse ms | Dense zero-filled ms |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0% | none | 5 | 0.0000 ± 0.0000 | 0.9989 ± 0.0025 | 0.0% | 1.416 | 0.053 |
| 50% | magnitude | 5 | 0.5000 ± 0.0000 | 0.9922 ± 0.0075 | 50.0% | 1.450 | 0.065 |
| 50% | taylor | 5 | 0.5000 ± 0.0000 | 0.9989 ± 0.0025 | 50.0% | 1.189 | 0.054 |
| 75% | magnitude | 5 | 0.7500 ± 0.0000 | 0.9944 ± 0.0039 | 75.0% | 1.190 | 0.075 |
| 75% | taylor | 5 | 0.7500 ± 0.0000 | 0.9922 ± 0.0063 | 75.0% | 1.280 | 0.072 |
| 90% | magnitude | 5 | 0.8999 ± 0.0000 | 0.9022 ± 0.0736 | 90.0% | 0.843 | 0.062 |
| 90% | taylor | 5 | 0.8999 ± 0.0000 | 0.9433 ± 0.0464 | 90.0% | 0.949 | 0.061 |
| 95% | magnitude | 5 | 0.9500 ± 0.0000 | 0.6956 ± 0.0401 | 95.0% | 0.712 | 0.065 |
| 95% | taylor | 5 | 0.9500 ± 0.0000 | 0.7789 ± 0.0803 | 95.0% | 0.959 | 0.074 |

## Quantitative claim at 90% target sparsity

Across 5 paired seeds at 90% target sparsity, Taylor test accuracy was 94.33% ± 4.64 percentage points, versus 90.22% ± 7.36 for magnitude pruning. The paired mean difference (Taylor − magnitude) was 4.11 percentage points (SD 7.63; approximate 95% normal interval [-2.58, 10.80]).

The approximate interval includes zero, so this sweep does not provide clear evidence that either method is better at this target. Treat the comparison as inconclusive, especially with a small number of seeds.

## Interpretation limits

- Multiply-count reduction is a structural work estimate: it counts only active weight multiplications on the sparse-aware path. It does not include all memory, indexing, bias, activation, or Python-loop costs.
- The sparse-aware implementation may be slower in wall-clock time on small networks despite doing fewer multiply operations. Report the measured timings as observed, not as a guaranteed deployment speedup.
- The approximate interval uses a normal 1.96 multiplier and is descriptive for this small repeated-seed experiment, not a definitive significance test.
- Do not claim that the synthetic spiral result generalizes to real-world datasets without additional experiments.

## Files

- `part4_results_raw.csv`: one row per seed/method/target.
- `part4_results_summary.csv`: mean and standard deviation by target and method.
- `part4_pareto.png`: final test accuracy versus achieved sparsity with seed standard-deviation error bars.
- `part4_report.md`: setup, results, cost measurements, and generated claim.
