"""A simple two-layer MLP classifier built on the custom Tensor engine."""
import numpy as np
from tensor import Tensor
from masked_parameter import MaskedParameter, MaskedSGD

class MLPClassifier:
    def __init__(self, input_dim, hidden_dim, num_classes, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = MaskedParameter(Tensor(rng.normal(
        0, np.sqrt(2.0 / input_dim),
        size=(input_dim, hidden_dim)), requires_grad=True))
        self.b1 = MaskedParameter(Tensor(np.zeros((1, hidden_dim)), requires_grad=True))
        self.W2 =  MaskedParameter(Tensor(rng.normal(
        0, np.sqrt(2.0 / hidden_dim),
        size=(hidden_dim, num_classes)), requires_grad=True))
        self.b2 = MaskedParameter(Tensor(np.zeros((1, num_classes)), requires_grad=True))
    def parameters(self): return [self.W1,self.b1,self.W2,self.b2]
    def __call__(self, x):
        x = x if isinstance(x, Tensor) else Tensor(x)

        hidden = (
            x @ self.W1.effective_tensor()
            + self.b1.effective_tensor()).relu()

        logits = (
            hidden @ self.W2.effective_tensor()
            + self.b2.effective_tensor()
        )

        return logits
    def predict(self, x): return np.argmax(self(x).data, axis=1)
    def accuracy(self, x, y): return float(np.mean(self.predict(x) == np.asarray(y)))
