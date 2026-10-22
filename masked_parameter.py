"""Mask-aware parameters and optimizers for the custom NumPy Tensor engine."""
import numpy as np
from tensor import Tensor


class MaskedParameter:
    """Tensor plus boolean mask (True=active, False=pruned)."""
    def __init__(self, data_or_tensor, requires_grad=True):
        self.tensor = data_or_tensor if isinstance(data_or_tensor, Tensor) else Tensor(data_or_tensor, requires_grad=requires_grad)
        self.mask = np.ones_like(self.tensor.data, dtype=bool)
        # Accumulate mask changes until every optimizer has observed a step.
        self._changed_since_step = np.zeros_like(self.mask, dtype=bool)

    @property
    def data(self): return self.tensor.data
    @property
    def grad(self): return self.tensor.grad
    @property
    def shape(self): return self.tensor.shape
    @property
    def requires_grad(self): return self.tensor.requires_grad

    def effective_tensor(self):
        """Differentiable W_eff = W * mask; inactive entries get zero gradient."""
        return self.tensor * self.mask.astype(np.float64)

    def set_mask(self, new_mask, reset_values=True):
        new_mask = np.asarray(new_mask, dtype=bool)
        if new_mask.shape != self.mask.shape:
            raise ValueError(f"mask shape {new_mask.shape} != parameter shape {self.mask.shape}")
        changed = new_mask != self.mask
        self._changed_since_step |= changed
        if reset_values:
            # Explicit policy: both newly pruned and newly revived entries start at zero.
            self.tensor.data[changed] = 0.0
        else:
            self.tensor.data[~new_mask] = 0.0
        self.mask = new_mask.copy()
        return changed

    def enforce_mask(self):
        self.tensor.data[~self.mask] = 0.0

    def zero_grad(self): self.tensor.zero_grad()


class MaskedSGD:
    """SGD/momentum; inactive values and velocity are zero, changed state resets."""
    def __init__(self, parameters, learning_rate=0.01, momentum=0.0):
        if learning_rate <= 0: raise ValueError("learning_rate must be positive")
        if not 0 <= momentum < 1: raise ValueError("momentum must be in [0, 1)")
        self.parameters = list(parameters)
        if not all(isinstance(p, MaskedParameter) for p in self.parameters):
            raise TypeError("MaskedSGD expects MaskedParameter objects")
        self.lr, self.momentum = float(learning_rate), float(momentum)
        self.velocity = [np.zeros_like(p.data) for p in self.parameters]

    def zero_grad(self):
        for p in self.parameters: p.zero_grad()

    def step(self):
        for i, p in enumerate(self.parameters):
            changed = p._changed_since_step
            self.velocity[i][changed] = 0.0
            if p.grad is not None:
                p.grad[~p.mask] = 0.0
                if self.momentum:
                    self.velocity[i] = self.momentum * self.velocity[i] + p.grad
                    self.velocity[i][~p.mask] = 0.0
                    update = self.velocity[i]
                else:
                    update = p.grad
                p.data[p.mask] -= self.lr * update[p.mask]
            p.enforce_mask()
            p._changed_since_step[:] = False


class MaskedAdam:
    """Adam with per-entry moments and bias-correction counters reset on mask changes."""
    def __init__(self, parameters, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        if learning_rate <= 0: raise ValueError("learning_rate must be positive")
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1: raise ValueError("betas must be in [0, 1)")
        if eps <= 0: raise ValueError("eps must be positive")
        self.parameters = list(parameters)
        if not all(isinstance(p, MaskedParameter) for p in self.parameters):
            raise TypeError("MaskedAdam expects MaskedParameter objects")
        self.lr, self.beta1, self.beta2, self.eps = float(learning_rate), beta1, beta2, eps
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]
        self.t = [np.zeros_like(p.data, dtype=np.int64) for p in self.parameters]

    def zero_grad(self):
        for p in self.parameters: p.zero_grad()

    def step(self):
        for i, p in enumerate(self.parameters):
            changed, active = p._changed_since_step, p.mask
            self.m[i][changed] = 0.0
            self.v[i][changed] = 0.0
            self.t[i][changed] = 0
            if p.grad is not None:
                p.grad[~active] = 0.0
                self.m[i][active] = self.beta1 * self.m[i][active] + (1-self.beta1) * p.grad[active]
                self.v[i][active] = self.beta2 * self.v[i][active] + (1-self.beta2) * p.grad[active]**2
                self.t[i][active] += 1
                t = self.t[i][active]
                mhat = self.m[i][active] / (1 - self.beta1**t)
                vhat = self.v[i][active] / (1 - self.beta2**t)
                p.data[active] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
            self.m[i][~active] = 0.0
            self.v[i][~active] = 0.0
            self.t[i][~active] = 0
            p.enforce_mask()
            p._changed_since_step[:] = False
