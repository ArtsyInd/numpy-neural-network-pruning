import numpy as np
from tensor import Tensor
from masked_parameter import MaskedParameter

class Linear:
    """Fully connected layer; optionally supports connection pruning."""
    def __init__(self, in_features, out_features, rng, initialization="he", prunable=False):
        if in_features <= 0 or out_features <= 0:
            raise ValueError("Layer dimensions must be positive")
        if initialization == "he":
            scale = np.sqrt(2.0 / in_features)
        elif initialization == "xavier":
            scale = np.sqrt(2.0 / (in_features + out_features))
        else:
            raise ValueError("Use 'he' or 'xavier' initialization")
        weight = rng.normal(0.0, scale, size=(in_features, out_features))
        self.prunable = bool(prunable)
        self.weight = MaskedParameter(Tensor(weight, True)) if self.prunable else Tensor(weight, True)
        self.bias = Tensor(np.zeros(out_features, dtype=np.float64), True)

    def __call__(self, x):
        weight = self.weight.effective_tensor() if self.prunable else self.weight
        return x @ weight + self.bias

    def parameters(self):
        return [self.weight.tensor if self.prunable else self.weight, self.bias]

    def masked_parameters(self):
        return [self.weight] if self.prunable else []

    def dense_parameters(self):
        return [self.bias]

class ReLU:
    def __call__(self, x): return x.relu()
    def parameters(self): return []
    def masked_parameters(self): return []
    def dense_parameters(self): return []

class Sequential:
    def __init__(self, *layers): self.layers = list(layers)
    def __call__(self, x):
        for layer in self.layers: x = layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    def masked_parameters(self):
        return [p for layer in self.layers for p in layer.masked_parameters()]
    def dense_parameters(self):
        return [p for layer in self.layers for p in layer.dense_parameters()]
    def prunable_layers(self):
        return [layer for layer in self.layers if isinstance(layer, Linear) and layer.prunable]