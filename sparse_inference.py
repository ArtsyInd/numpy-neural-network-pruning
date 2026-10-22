"""Sparse-aware NumPy inference that skips inactive connections."""
import numpy as np

def sparse_linear(x, weight, mask, bias):
    x, weight = np.asarray(x, float), np.asarray(weight, float)
    mask, bias = np.asarray(mask, bool), np.asarray(bias, float)
    if weight.shape != mask.shape: raise ValueError("weight and mask shapes must match")
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[0]:
        raise ValueError("incompatible input/weight shapes")
    if bias.shape != (weight.shape[1],): raise ValueError("invalid bias shape")
    out = np.broadcast_to(bias, (x.shape[0], bias.size)).copy()
    multiplies = 0
    for j in range(weight.shape[1]):
        idx = np.flatnonzero(mask[:, j])
        if idx.size:
            out[:, j] += x[:, idx] @ weight[idx, j]
            multiplies += x.shape[0] * idx.size
    return out, int(multiplies)

def predict_sparse(model, x):
    from layers import Linear, ReLU
    x = np.asarray(x, dtype=np.float64); multiplies = 0
    for layer in model.layers:
        if isinstance(layer, Linear):
            if layer.prunable:
                x, n = sparse_linear(x, layer.weight.data, layer.weight.mask, layer.bias.data)
                multiplies += n
            else:
                x = x @ layer.weight.data + layer.bias.data
                multiplies += x.shape[0] * layer.weight.data.size
        elif isinstance(layer, ReLU): x = np.maximum(x, 0.0)
        else: raise TypeError(f"Unsupported layer: {type(layer).__name__}")
    return x, multiplies

def dense_masked_predict(model, x):
    from layers import Linear, ReLU
    x = np.asarray(x, dtype=np.float64)
    for layer in model.layers:
        if isinstance(layer, Linear):
            w = layer.weight.data if not layer.prunable else layer.weight.data * layer.weight.mask
            x = x @ w + layer.bias.data
        elif isinstance(layer, ReLU): x = np.maximum(x, 0.0)
        else: raise TypeError(f"Unsupported layer: {type(layer).__name__}")
    return x
def dense_zero_predict(model, x):
    """Dense baseline using full matrix multiplies over exact-zero pruned weights.

    Unlike `predict_sparse`, this does not skip inactive connections. Because
    masked parameters enforce exact zeros, it can use stored weights directly.
    """
    from layers import Linear, ReLU
    x = np.asarray(x, dtype=np.float64)
    for layer in model.layers:
        if isinstance(layer, Linear):
            # Inactive entries are guaranteed to be exactly zero by the optimizer.
            x = x @ layer.weight.data + layer.bias.data
        elif isinstance(layer, ReLU):
            x = np.maximum(x, 0.0)
        else:
            raise TypeError(f"Unsupported layer: {type(layer).__name__}")
    return x

