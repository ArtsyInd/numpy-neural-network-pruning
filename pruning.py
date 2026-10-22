"""Mask-aware parameters and Adam optimizer for the NumPy autodiff Tensor."""
import numpy as np
from tensor import Tensor


class MaskedParameter:
    """
    Wrap a trainable Tensor with a binary connectivity mask.

    `value` stores the underlying parameter. `effective()` returns the value
    used by the forward pass. Multiplication by the mask makes autodiff produce
    zero gradients for masked entries.
    """
    def __init__(self, value, mask=None):
        if not isinstance(value, Tensor) or not value.requires_grad:
            raise TypeError("value must be a trainable Tensor")
        self.value = value
        if mask is None:
            mask = np.ones_like(value.data)
        mask = np.asarray(mask, dtype=np.float64)
        if mask.shape != value.shape:
            raise ValueError("mask shape must match parameter shape")
        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError("mask must contain only 0 and 1")
        self.mask = mask.copy()
        self.value.data *= self.mask

    def effective(self):
        """Return the masked Tensor used in forward computation."""
        return self.value * self.mask


class MaskedAdam:
    """
    Adam optimizer that enforces masks and clears per-entry state whenever
    a connection changes active status (pruned OR revived).

    This implementation intentionally resets m and v for changed entries.
    This avoids stale momentum/adaptive moments affecting a revived connection.
    """
    def __init__(self, parameters, learning_rate=1e-3, beta1=0.9,
                 beta2=0.999, eps=1e-8):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not (0 <= beta1 < 1 and 0 <= beta2 < 1):
            raise ValueError("beta1 and beta2 must be in [0, 1)")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.parameters = list(parameters)
        self.learning_rate = float(learning_rate)
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.m = {id(p): np.zeros_like(p.value.data) for p in self.parameters}
        self.v = {id(p): np.zeros_like(p.value.data) for p in self.parameters}
        # Per-connection age is reset on prune/revival, so bias correction for a
        # revived connection does not inherit the age of its previous life.
        self.age = {id(p): np.zeros_like(p.value.data, dtype=np.int64)
                    for p in self.parameters}

    def zero_grad(self):
        for p in self.parameters:
            p.value.zero_grad()

    def set_mask(self, parameter, new_mask):
        """Change a parameter's mask and reset Adam state for changed entries."""
        if parameter not in self.parameters:
            raise ValueError("parameter is not managed by this optimizer")
        new_mask = np.asarray(new_mask, dtype=np.float64)
        if new_mask.shape != parameter.value.shape:
            raise ValueError("mask shape must match parameter shape")
        if not np.all((new_mask == 0) | (new_mask == 1)):
            raise ValueError("mask must contain only 0 and 1")

        key = id(parameter)
        changed = new_mask != parameter.mask
        # Clear state for both pruned and revived entries. In particular, a
        # revived connection starts with no stale momentum from before pruning.
        self.m[key][changed] = 0.0
        self.v[key][changed] = 0.0
        self.age[key][changed] = 0
        parameter.mask = new_mask.copy()

        # Enforce the invariant immediately: inactive raw weights are zero.
        parameter.value.data *= parameter.mask
        parameter.value.grad *= parameter.mask

    def step(self):
        for p in self.parameters:
            key = id(p)
            mask = p.mask
            grad = p.value.grad * mask

            # Never let inactive entries accumulate optimizer state.
            self.m[key] *= mask
            self.v[key] *= mask
            self.m[key] = self.beta1 * self.m[key] + (1.0 - self.beta1) * grad
            self.v[key] = self.beta2 * self.v[key] + (1.0 - self.beta2) * (grad * grad)

            active = mask.astype(bool)
            self.age[key][active] += 1
            age = self.age[key]
            # Per-entry bias correction. Inactive entries have age 0; use a
            # safe denominator there, then the mask makes their update zero.
            b1_correction = 1.0 - np.power(self.beta1, np.maximum(age, 1))
            b2_correction = 1.0 - np.power(self.beta2, np.maximum(age, 1))
            m_hat = self.m[key] / b1_correction
            v_hat = self.v[key] / b2_correction
            update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

            p.value.data -= update * mask
            # Hard invariant: inactive weights remain exactly zero.
            p.value.data *= mask


def cubic_sparsity(step, start_step, end_step, final_sparsity):
    if not 0.0 <= final_sparsity < 1.0:
        raise ValueError("final_sparsity must be in [0, 1)")
    if end_step <= start_step:
        raise ValueError("end_step must be greater than start_step")
    if step <= start_step: return 0.0
    if step >= end_step: return float(final_sparsity)
    progress = (step - start_step) / (end_step - start_step)
    return float(final_sparsity * (1.0 - (1.0 - progress) ** 3))

def magnitude_scores(parameter):
    return np.abs(parameter.data)

def taylor_scores(parameter):
    if parameter.grad is None:
        raise ValueError("Gradient unavailable; run backward() before Taylor scoring")
    return np.abs(parameter.data * parameter.grad)

class GlobalPruner:
    """Prune globally across MaskedParameters; biases are not passed to this class."""
    def __init__(self, parameters, method="taylor"):
        self.parameters = list(parameters)
        if not self.parameters: raise ValueError("At least one parameter is required")
        if method not in {"taylor", "magnitude"}:
            raise ValueError("method must be 'taylor' or 'magnitude'")
        self.method = method

    def _score(self, p):
        return taylor_scores(p) if self.method == "taylor" else magnitude_scores(p)

    def prune_to_sparsity(self, target_sparsity):
        if not 0.0 <= target_sparsity < 1.0:
            raise ValueError("target_sparsity must be in [0, 1)")
        total = sum(p.mask.size for p in self.parameters)
        target_pruned = int(round(target_sparsity * total))
        current_pruned = sum(int((~p.mask).sum()) for p in self.parameters)
        count = max(0, min(target_pruned-current_pruned, total-current_pruned))
        if count == 0: return self.statistics()
        chunks = []
        for p in self.parameters:
            score = np.asarray(self._score(p), dtype=np.float64)
            chunks.append(np.where(p.mask, score, np.inf).ravel())
        all_scores = np.concatenate(chunks)
        indices = np.argsort(all_scores, kind="stable")[:count]
        offset = 0
        for p in self.parameters:
            size = p.mask.size
            local = indices[(indices >= offset) & (indices < offset+size)] - offset
            mask = p.mask.copy().ravel()
            mask[local] = False
            p.set_mask(mask.reshape(p.mask.shape), reset_values=True)
            offset += size
        return self.statistics()

    def statistics(self):
        total = sum(p.mask.size for p in self.parameters)
        active = sum(int(p.mask.sum()) for p in self.parameters)
        return {"total_connections": total, "active_connections": active,
                "pruned_connections": total-active,
                "sparsity": (total-active)/total if total else 0.0}
