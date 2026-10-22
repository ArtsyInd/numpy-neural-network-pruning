# Design Notes

This document explains the main decisions behind the NumPy neural-network implementation, how I handle pruning and gradients, and what I learned from testing the sparse inference path.

## 1. Why I used a Taylor-based importance score

Magnitude pruning removes weights with the smallest absolute values. It is a useful baseline, but weight size alone does not tell us how much a connection contributes to the loss.

For this project, I also implemented a first-order Taylor-based score.

Let the model loss be L(w), where w_i is one weight. Removing that connection sets the weight to zero, so the parameter change is:

Delta w_i = -w_i

Using a first-order Taylor approximation:

L(w + Delta w_i) ≈ L(w) + (dL/dw_i) * Delta w_i

Substituting Delta w_i = -w_i gives:

Delta L_i ≈ -w_i * (dL/dw_i)

I use the absolute value of this estimate as the importance score:

I_i = |w_i * (dL/dw_i)|

A smaller score suggests that removing the connection should have a smaller first-order effect on the loss. The score is a heuristic, not a guarantee that the loss will remain unchanged.

The approximation ignores higher-order terms and interactions between weights. The result can also depend on the batch used to calculate the gradients. These limitations matter when interpreting the results, especially at high sparsity.

I compare this method with magnitude pruning, which ranks weights using \(|w_i|\) without considering their gradients. Comparing both methods at the same target sparsity helps isolate differences in the pruning criterion.

## 2. What happens to gradients when a weight is masked?

Let m_i be a binary mask: 1 means active and 0 means pruned. The effective weight used in the forward pass is:

effective_w_i = m_i * w_i

By the chain rule:

dL/dw_i = (dL/d(effective_w_i)) * (d(effective_w_i)/dw_i)

Since d(effective_w_i)/dw_i = m_i:

dL/dw_i = m_i * dL/d(effective_w_i)

When m_i = 1, the underlying weight receives its usual gradient. When m_i = 0, its gradient is zero because changing that weight cannot affect the forward computation while the mask remains zero.
There is another issue beyond the gradient itself: optimizer state.

Adam maintains moving estimates of gradients and squared gradients. Simply zeroing a weight or its gradient is not enough to guarantee that old optimizer state cannot affect it. My implementation therefore enforces the mask during optimization and resets the relevant optimizer state when a mask entry changes.

This also matters if a connection is reactivated. Resetting its state prevents momentum or variance accumulated under the previous mask from carrying over unexpectedly.

The important distinction is that the mask determines whether a connection participates in the forward computation. A pruned weight should remain exactly zero, and optimizer updates should not silently reactivate it.

## 3. Autodiff design and performance trade-offs

The engine uses NumPy operations and stores intermediate tensors and gradient information for backpropagation. Potential bottlenecks include allocating temporary arrays, traversing the computation graph, and repeatedly processing small operations in Python.

Possible optimizations include reusing buffers, reducing unnecessary copies, improving graph traversal, and combining compatible operations. Profiling should determine which changes matter most. For larger models, vectorized kernels or an established numerical backend could reduce overhead.

## 4. Multi-Tenant Production Serving

A production service should separate offline model preparation from online inference.

Offline, each model version would be trained, pruned, evaluated, and stored with its weights, masks, configuration, and validation results. Models would be published only after meeting accuracy and resource requirements.

Online, requests would be routed to the correct tenant and model version. Tenant-specific configuration, access controls, resource limits, and monitoring would prevent one tenant from affecting another. Model artifacts should be immutable and versioned to support rollback.

The serving format should match the available hardware. Zero-filled dense weights do not automatically provide faster inference; a sparse representation and compatible kernels are needed to benefit from skipped connections. The current project demonstrates sparse-aware inference, but a scalable multi-tenant service remains a proposed production extension.

## 5. Experimental Evidence

The Pareto experiment compares magnitude and Taylor pruning at target sparsities of 50%, 75%, 90%, and 95%, using five random seeds and an unpruned baseline.

At 90% sparsity, mean test accuracy was 94.33% for Taylor pruning and 90.22% for magnitude pruning. However, the variation across seeds means this result is inconclusive evidence of superiority. At higher sparsity, both methods lose accuracy.

Sparse-aware inference reduced the number of multiplications, but was not consistently faster than dense inference on this small network. The results therefore demonstrate a sparsity–accuracy trade-off, not a general inference-speedup guarantee.
