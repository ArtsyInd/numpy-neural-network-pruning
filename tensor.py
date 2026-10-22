"""A small reverse-mode autodiff engine built only with Python and NumPy."""
import numpy as np

def _unbroadcast(grad, target_shape):
    grad = np.asarray(grad, dtype=np.float64)
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(target_shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(target_shape)

class Tensor:
    """NumPy-backed tensor with a reverse-mode computation graph."""
    __array_priority__ = 1000
    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = bool(requires_grad)
        self.grad = np.zeros_like(self.data) if self.requires_grad else None
        self._prev = tuple(_children)
        self._op = _op
        self._backward = lambda: None
    @property
    def shape(self): return self.data.shape
    @property
    def ndim(self): return self.data.ndim
    def __repr__(self):
        return f"Tensor(data={self.data!r}, requires_grad={self.requires_grad})"
    def _binary(self, other, name, forward, backward):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(forward(self.data, other.data),
                     self.requires_grad or other.requires_grad, (self, other), name)
        def _backward():
            if out.grad is None: return
            ga, gb = backward(self.data, other.data, out.data, out.grad)
            if self.requires_grad: self.grad += _unbroadcast(ga, self.shape)
            if other.requires_grad: other.grad += _unbroadcast(gb, other.shape)
        out._backward = _backward
        return out
    def __add__(self, other):
        return self._binary(other, "add", np.add, lambda a,b,o,g: (g,g))
    def __radd__(self, other): return self + other
    def __sub__(self, other):
        return self._binary(other, "sub", np.subtract, lambda a,b,o,g: (g,-g))
    def __rsub__(self, other): return Tensor(other) - self
    def __mul__(self, other):
        return self._binary(other, "mul", np.multiply, lambda a,b,o,g: (g*b,g*a))
    def __rmul__(self, other): return self * other
    def __truediv__(self, other):
        return self._binary(other, "div", np.divide,
            lambda a,b,o,g: (g/b, -g*a/(b*b)))
    def __rtruediv__(self, other): return Tensor(other) / self
    def __neg__(self): return self * -1.0
    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        if self.ndim != 2 or other.ndim != 2:
            raise ValueError("This minimal engine supports 2-D matrix multiplication only.")
        out = Tensor(self.data @ other.data,
                     self.requires_grad or other.requires_grad, (self,other), "matmul")
        def _backward():
            if self.requires_grad: self.grad += out.grad @ other.data.T
            if other.requires_grad: other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                     self.requires_grad, (self,), "sum")
        def _backward():
            if not self.requires_grad: return
            grad = out.grad
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis,int) else tuple(axis)
                axes = tuple(a if a >= 0 else a+self.ndim for a in axes)
                for a in sorted(axes): grad = np.expand_dims(grad, axis=a)
            self.grad += np.ones_like(self.data) * grad
        out._backward = _backward
        return out
    def mean(self, axis=None, keepdims=False):
        if axis is None: count = self.data.size
        else:
            axes = (axis,) if isinstance(axis,int) else tuple(axis)
            axes = tuple(a if a >= 0 else a+self.ndim for a in axes)
            count = int(np.prod([self.shape[a] for a in axes]))
        return self.sum(axis=axis, keepdims=keepdims) / count
    def relu(self):
        value = np.maximum(self.data, 0.0)
        out = Tensor(value, self.requires_grad, (self,), "relu")
        def _backward():
            if self.requires_grad: self.grad += out.grad * (self.data > 0.0)
        out._backward = _backward
        return out
    def tanh(self):
        value = np.tanh(self.data)
        out = Tensor(value, self.requires_grad, (self,), "tanh")
        def _backward():
            if self.requires_grad: self.grad += out.grad * (1.0-value*value)
        out._backward = _backward
        return out
    
    def sigmoid(self):
        x = self.data.astype(np.float64, copy=False)

        # Numerically stable sigmoid
        value = np.empty_like(x, dtype=np.float64)
        positive = x >= 0

        value[positive] = 1.0 / (1.0 + np.exp(-x[positive]))

        exp_x = np.exp(x[~positive])
        value[~positive] = exp_x / (1.0 + exp_x)

        out = Tensor(value, self.requires_grad, (self,), "sigmoid")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * value * (1.0 - value)

        out._backward = _backward
        return out
        
    def gelu(self):
        x = self.data.astype(np.float64, copy=False)

        # Tanh approximation of GELU
        a = np.sqrt(2.0 / np.pi)
        c = 0.044715

        u = a * (x + c * x**3)
        t = np.tanh(u)
        value = 0.5 * x * (1.0 + t)

        out = Tensor(value, self.requires_grad, (self,), "gelu")

        def _backward():
            if self.requires_grad:
                du_dx = a * (1.0 + 3.0 * c * x**2)

                derivative = (
                    0.5 * (1.0 + t)
                    + 0.5 * x * (1.0 - t**2) * du_dx
                )

                self.grad += out.grad * derivative

        out._backward = _backward
        return out

    def exp(self):
        value = np.exp(self.data)
        out = Tensor(value, self.requires_grad, (self,), "exp")
        def _backward():
            if self.requires_grad: self.grad += out.grad * value
        out._backward = _backward
        return out
    def log(self):
        out = Tensor(np.log(self.data), self.requires_grad, (self,), "log")
        def _backward():
            if self.requires_grad: self.grad += out.grad / self.data
        out._backward = _backward
        return out
    def softmax_cross_entropy(self, labels):
        """Stable mean softmax cross-entropy from logits; labels are class indices."""
        if self.ndim != 2: raise ValueError("logits must have shape (batch, classes)")
        labels = np.asarray(labels, dtype=np.int64)
        n, classes = self.shape
        if n == 0: raise ValueError("batch must not be empty")
        if labels.shape != (n,): raise ValueError("labels must have shape (batch,)")
        if np.any(labels < 0) or np.any(labels >= classes):
            raise ValueError("labels contain an invalid class index")
        shifted = self.data - self.data.max(axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        loss_value = -np.log(np.maximum(probs[np.arange(n),labels], 1e-300)).mean()
        out = Tensor(loss_value, self.requires_grad, (self,), "softmax_cross_entropy")
        def _backward():
            if self.requires_grad:
                grad = probs.copy()
                grad[np.arange(n),labels] -= 1.0
                self.grad += grad * (out.grad / n)
        out._backward = _backward
        return out
    def backward(self, grad=None):
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward() on a tensor that does not require gradients")
        if grad is None:
            if self.data.size != 1: raise ValueError("Provide grad for a non-scalar output")
            grad = np.ones_like(self.data)
        grad = np.asarray(grad, dtype=np.float64)
        if grad.shape != self.shape:
            raise ValueError(f"grad shape {grad.shape} does not match output shape {self.shape}")
        self.grad += grad
        topo, visited = [], set()
        def visit(node):
            if id(node) in visited: return
            visited.add(id(node))
            for parent in node._prev: visit(parent)
            topo.append(node)
        visit(self)
        for node in reversed(topo): node._backward()
    def zero_grad(self):
        if self.requires_grad: self.grad.fill(0.0)
