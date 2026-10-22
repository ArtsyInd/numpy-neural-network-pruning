"""Basic SGD optimizer implemented without third-party ML libraries."""
import numpy as np

class SGD:
    def __init__(self, parameters, learning_rate=0.05):
        if learning_rate <= 0: raise ValueError("learning_rate must be positive")
        self.parameters = list(parameters)
        self.learning_rate = float(learning_rate)
    def step(self):
        for p in self.parameters:
            if p.requires_grad: p.data -= self.learning_rate * p.grad
    def zero_grad(self):
        for p in self.parameters: p.zero_grad()
class Adam:
    """Adam optimizer implemented using NumPy."""

    def __init__(
        self,
        parameters,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
    ):
        self.parameters = list(parameters)

        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= beta1 < 1:
            raise ValueError("beta1 must be in [0, 1)")
        if not 0 <= beta2 < 1:
            raise ValueError("beta2 must be in [0, 1)")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()

    def step(self):
        self.t += 1

        correction1 = 1.0 - self.beta1**self.t
        correction2 = 1.0 - self.beta2**self.t

        for i, p in enumerate(self.parameters):
            if not p.requires_grad:
                continue

            grad = np.asarray(p.grad)

            if not np.all(np.isfinite(grad)):
                raise FloatingPointError(
                    f"Non-finite gradient in parameter {i}"
                )

            self.m[i] = (
                self.beta1 * self.m[i]
                + (1.0 - self.beta1) * grad
            )
            self.v[i] = (
                self.beta2 * self.v[i]
                + (1.0 - self.beta2) * (grad * grad)
            )

            m_hat = self.m[i] / correction1
            v_hat = self.v[i] / correction2

            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            if not np.all(np.isfinite(update)):
                raise FloatingPointError(
                    f"Non-finite Adam update in parameter {i}"
                )

            p.data -= update

            if not np.all(np.isfinite(p.data)):
                raise FloatingPointError(
                    f"Non-finite parameter values in parameter {i}"
                )

