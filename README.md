# NumPy Neural Network and Self-Pruning Engine

A small neural-network framework implemented with Python and NumPy. The project covers reverse-mode automatic differentiation, an MLP and optimizer implemented from scratch, masked weights, self-pruning, and sparsity–accuracy evaluation.

## Features

- **Part 1 — Autodiff:** tensor computation graph, reverse-mode backpropagation, gradient accumulation, broadcasting-aware gradient reduction, stable classification loss, and finite-difference gradient checks.
- **Part 2 — Neural network:** composable fully connected layers and activations, NumPy Adam, mini-batch training, and a synthetic three-class spiral classification example.
- **Part 3 — Self-pruning:** binary weight masks, gradual cubic sparsity schedule, magnitude and Taylor importance scores, masked optimizers, optimizer-state resets when mask entries change, and sparse-aware inference.
- **Part 4 — Evaluation:** multi-seed sparsity sweep, raw and aggregated CSV files, Pareto plot, and a report comparing accuracy with structural and measured inference costs.

## Requirements and installation

Python 3.10+ is recommended. The core engine uses NumPy and the standard library; Matplotlib is used to create plots. No PyTorch, TensorFlow, JAX, Keras, or third-party autodiff/pruning library is used.

### With `uv`

Install [uv](https://docs.astral.sh/uv/), then from the repository root:

```bash
uv venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
uv pip install -r requirements.txt
```

### With `pip`

**Windows PowerShell**
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**macOS/Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Tests

Run the complete test suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

Run the Part 1 gradient-check suite:

```bash
python -m unittest discover -s tests -p "test_gradient_checks.py" -v
```

The tests cover finite-difference gradient checks and masked-weight behavior, along with optimizer, pruning, and sparse-inference tests. 

### Part 2 — Baseline MLP training

```bash
python train_part2.py
```

### Part 3 — Self-pruning training

Taylor importance:

```bash
python train_part3.py --epochs 300 --target-sparsity 0.80 --prune-method taylor
```

Magnitude baseline:

```bash
python train_part3.py --epochs 300 --target-sparsity 0.80 --prune-method magnitude --output-prefix part3_magnitude
```

### Part 4 — Full Pareto sweep

```bash
python run_pareto.py --epochs 300 --targets 0,0.5,0.75,0.9,0.95 --seeds 11,22,33,44,55 --timing-repeats 7 --output-dir part4_results
```

The sweep runs an unpruned baseline for each seed and both pruning methods at four nonzero sparsity targets. It writes per-run and summary CSVs, a plot, and a Markdown report into `part4_results/`.

Optional quick smoke test:

```bash
python run_pareto.py --epochs 10 --targets 0,0.5 --seeds 11 --prune-start-fraction 0.1 --prune-end-fraction 0.8 --prune-every 1 --timing-repeats 2 --output-dir part4_smoke
```

### Findings from the reported run

At 90% target sparsity, Taylor pruning achieved **94.33% ± 4.64 percentage points** mean test accuracy, compared with **90.22% ± 7.36 percentage points** for magnitude pruning across five seeds. The paired mean difference was 4.11 percentage points, but its approximate 95% interval included zero; this experiment therefore does not establish that Taylor pruning is better.

At 95% sparsity, accuracy dropped for both methods. The sparse-aware path reduced the counted active weight multiplications, but measured wall-clock latency was higher than the dense zero-filled baseline on this small network. This is a limitation of the current Python implementation, not evidence of a deployment speedup.

These results use a synthetic spiral dataset and should not be generalized to real-world tasks without further experiments. Latency depends on hardware, system load, batch size, and implementation details.

## Design document

See [DESIGN.md](DESIGN.md) for the Taylor criterion derivation, masked-gradient semantics, autodiff bottlenecks, and a proposed multi-tenant inference architecture.

## Reproducibility and limitations

- Standardization should be fitted on training data only.
- Five seeds provide an initial estimate of variability, not strong evidence for small differences.
- Multiply-count reduction is a structural work estimate, not a wall-clock speedup guarantee.
- Python-level sparse loops can be slower than dense matrix multiplication for small networks. A production implementation should benchmark appropriate sparse formats or compiled kernels on target hardware.
- Rerun tests and the full sweep on the final checkout, then commit the outputs generated by that run.

## License

This project is licensed under the MIT License.
