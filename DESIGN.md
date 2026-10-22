# Design Notes

## 1. Importance criterion: Taylor approximation

Let a trained model have loss \(L(\mathbf{w})\). Removing a connection with weight \(w_i\) sets it to zero, so the parameter perturbation is \(\Delta w_i=-w_i\). A first-order Taylor expansion gives

\[
L(\mathbf{w}+\Delta w_i) \approx L(\mathbf{w}) + \frac{\partial L}{\partial w_i}\Delta w_i.
\]

Therefore,

\[
\Delta L_i \approx -w_i\frac{\partial L}{\partial w_i}.
\]

The signed expression estimates the first-order loss change. For ranking pruning candidates by the magnitude of the estimated effect, use

\[
I_i = \left|w_i\frac{\partial L}{\partial w_i}\right|.
\]

Connections with smaller scores are considered less consequential under this local approximation. This is a heuristic, not a guarantee: it ignores higher-order terms, interactions among removed connections, and later adaptation during training. Scores can also be noisy if calculated on a single mini-batch. Magnitude pruning instead ranks by \(|w_i|\) and does not account for the loss gradient.

## 2. Gradient of a masked weight

Let \(m_i\in\{0,1\}\) be a binary mask and let the effective weight be \(\tilde w_i=m_iw_i\). The forward pass uses the effective weight. By the chain rule,

\[
\frac{\partial L}{\partial w_i} = \frac{\partial L}{\partial \tilde w_i}\frac{\partial \tilde w_i}{\partial w_i} = m_i\frac{\partial L}{\partial \tilde w_i}.
\]

For an active connection, the underlying parameter receives the usual gradient. For a masked connection, the gradient with respect to the underlying weight is zero. This is consistent with the forward computation: while the mask is zero, changing that weight cannot change the output or loss. The implementation also enforces exact zero values and prevents optimizers from updating inactive connections. When a mask entry changes, its optimizer state is reset so momentum/variance accumulated under the old mask does not unexpectedly affect the newly active or inactive connection. If a connection is revived, it can receive gradients again after its mask is set to one.

## 3. Autodiff bottlenecks and optimizations

This engine prioritizes transparency over throughput. Likely bottlenecks include:

- **Python graph traversal:** each operation creates a tensor node and backward callback; graph ordering and callback dispatch add interpreter overhead.
- **Many small NumPy operations:** elementwise operations and gradient accumulation can allocate temporary arrays and make many calls instead of using fused kernels.
- **Gradient storage:** each differentiable tensor may hold a gradient buffer, increasing memory use for large intermediate activations.
- **Dense matrix multiplication:** NumPy's dense `matmul` is efficient for dense layers but does not automatically exploit zero-valued masked weights.
- **Sparse inference loops:** Python-level iteration and indexing can cost more than the multiplications saved, especially for small batches and networks.
- **Finite-difference tests:** numerical gradient checks require repeated forward evaluations and are intended for tests, not training.

Potential improvements should be guided by profiling:

1. Profile representative training and inference workloads before optimizing.
2. Reduce temporary allocations and reuse buffers where safe.
3. If callback overhead is significant, represent operations with compact metadata rather than a separate closure per operation.
4. Add vectorized or fused operations and validate their backward rules with numerical gradient tests.
5. Consider a compiled backend or generated kernels for the computation graph.
6. Use sparse matrix formats or compiled sparse kernels for inference, selecting formats based on layer shape, sparsity pattern, batch size, and target hardware.
7. Add numerical and performance regression tests after execution changes.

These are proposed optimizations, not features claimed to be implemented in this repository.

## 4. Serving a self-pruned model in a multi-tenant inference service

Training and pruning should be separated from online serving.

### Offline model build

1. Train and prune with versioned data, configuration, random seed, and code revision.
2. Validate quality, achieved sparsity, numerical equivalence, and any task-specific calibration requirements.
3. Export immutable artifacts containing weights, masks or sparse matrices, architecture metadata, preprocessing parameters, and a model/version identifier.
4. Register pruning method, target and achieved sparsity, validation metrics, software versions, and artifact checksum.
5. Benchmark end-to-end latency and memory at realistic batch sizes on the actual serving hardware before promotion.

### Online request path

- Authenticate the request and resolve the tenant's authorized model version through a tenant-to-model registry.
- Validate the input schema and apply the preprocessing version associated with that model.
- Route to workers with the model loaded; batch compatible requests when latency and isolation requirements permit.
- Return predictions with request and model-version metadata, avoiding unnecessary logging of sensitive payloads.

### Isolation, scaling, and reliability

- Enforce tenant authorization at the API and registry layers; do not trust a client-supplied model identifier without checking access.
- Apply per-tenant quotas, rate limits, concurrency limits, and bounded queues to prevent noisy-neighbor effects.
- Cache immutable model artifacts by version and use controlled warm-up and eviction policies.
- Use health checks, timeouts, safe retry policies, autoscaling, and gradual rollouts with rollback support.
- Monitor latency percentiles, throughput, errors, memory, queue depth, model version, and prediction/quality drift where labels or suitable feedback are available.
- Restrict model and tenant-data access, encrypt data in transit and at rest, and maintain appropriate audit trails.
- For stricter isolation or regulatory requirements, use dedicated workers or separate deployments rather than relying only on tenant IDs in application code.

### Sparse representation choice

A binary mask over dense weights guarantees zeroed connections but does not, by itself, reduce dense storage or dense matrix-multiplication cost. If inference speed or memory is a goal, export an actual sparse representation (for example, CSR/CSC for compatible workloads, or structured/block sparsity supported by target hardware). Benchmark the entire path, including conversion, memory movement, batching, and request overhead. The best representation depends on sparsity pattern, dimensions, batch size, hardware, and available kernels.

This repository is a single-process NumPy experiment and does not implement a production serving service. This section describes a plausible production design, not deployed functionality.
