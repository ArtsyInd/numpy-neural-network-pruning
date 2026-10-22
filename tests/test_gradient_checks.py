import unittest
import numpy as np

from tensor import Tensor


class TestFiniteDifferenceGradients(unittest.TestCase):
    # Central finite differences are accurate for smooth functions
    # when h is small but not too small.
    H = 1e-5
    RTOL = 2e-4
    ATOL = 2e-5

    def assert_gradients_match(self, function, inputs):
        """
        function: accepts a list of Tensors and returns a scalar Tensor.
        inputs: list of NumPy arrays used to create those Tensors.

        Compares autodiff gradients with central finite differences.
        """
        tensors = [
            Tensor(np.array(x, dtype=np.float64), requires_grad=True)
            for x in inputs
        ]

        loss = function(tensors)
        self.assertEqual(
            np.asarray(loss.data).size, 1,
            "Gradient checking requires a scalar output."
        )
        loss.backward()

        analytical = [t.grad.copy() for t in tensors]
        numerical = []

        for input_index, original in enumerate(inputs):
            grad = np.zeros_like(original, dtype=np.float64)

            for index in np.ndindex(original.shape):
                plus_inputs = [np.array(x, dtype=np.float64, copy=True)
                               for x in inputs]
                minus_inputs = [np.array(x, dtype=np.float64, copy=True)
                                for x in inputs]

                plus_inputs[input_index][index] += self.H
                minus_inputs[input_index][index] -= self.H

                plus_tensors = [Tensor(x) for x in plus_inputs]
                minus_tensors = [Tensor(x) for x in minus_inputs]

                plus_loss = function(plus_tensors)
                minus_loss = function(minus_tensors)

                plus_value = float(np.asarray(plus_loss.data))
                minus_value = float(np.asarray(minus_loss.data))

                grad[index] = (
                    plus_value - minus_value
                ) / (2.0 * self.H)

            numerical.append(grad)

        for i, (actual, expected) in enumerate(
            zip(analytical, numerical)
        ):
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=self.RTOL,
                atol=self.ATOL,
                err_msg=f"Gradient mismatch for input {i}",
            )

    def test_add(self):
        self.assert_gradients_match(
            lambda v: (v[0] + v[1]).sum(),
            [np.array([0.3, -0.7]), np.array([0.8, 0.2])],
        )

    def test_sub(self):
        self.assert_gradients_match(
            lambda v: (v[0] - v[1]).sum(),
            [np.array([0.3, -0.7]), np.array([0.8, 0.2])],
        )

    def test_mul(self):
        self.assert_gradients_match(
            lambda v: (v[0] * v[1]).sum(),
            [np.array([0.3, -0.7]), np.array([0.8, 0.2])],
        )

    def test_div(self):
        self.assert_gradients_match(
            lambda v: (v[0] / v[1]).sum(),
            [np.array([0.3, -0.7]), np.array([1.2, 0.9])],
        )

    def test_reverse_div(self):
        self.assert_gradients_match(
            lambda v: (2.0 / v[0]).sum(),
            [np.array([1.2, 0.9])],
        )

    def test_matmul(self):
        self.assert_gradients_match(
            lambda v: (v[0] @ v[1]).sum(),
            [
                np.array([[0.2, -0.4], [0.7, 0.5]]),
                np.array([[0.6, 0.1], [-0.3, 0.8]]),
            ],
        )

    def test_broadcast_add(self):
        self.assert_gradients_match(
            lambda v: (v[0] + v[1]).sum(),
            [
                np.array([[0.2, 0.4], [0.6, 0.8], [1.0, 1.2]]),
                np.array([0.1, 0.3]),
            ],
        )

    def test_sum(self):
        self.assert_gradients_match(
            lambda v: v[0].sum(),
            [np.array([[0.2, -0.4], [0.7, 0.5]])],
        )

    def test_mean(self):
        self.assert_gradients_match(
            lambda v: v[0].mean(),
            [np.array([[0.2, -0.4], [0.7, 0.5]])],
        )

    def test_exp(self):
        self.assert_gradients_match(
            lambda v: v[0].exp().sum(),
            [np.array([-0.5, 0.2, 0.8])],
        )

    def test_log(self):
        self.assert_gradients_match(
            lambda v: v[0].log().sum(),
            [np.array([0.5, 1.2, 2.0])],
        )

    def test_relu(self):
        # Avoid zero because ReLU is not differentiable at zero.
        self.assert_gradients_match(
            lambda v: v[0].relu().sum(),
            [np.array([-1.2, -0.3, 0.4, 1.5])],
        )

    def test_tanh(self):
        self.assert_gradients_match(
            lambda v: v[0].tanh().sum(),
            [np.array([-1.0, -0.2, 0.4, 1.1])],
        )

    def test_sigmoid(self):
        self.assert_gradients_match(
            lambda v: v[0].sigmoid().sum(),
            [np.array([-2.0, -0.4, 0.3, 1.5])],
        )

    def test_gelu(self):
        self.assert_gradients_match(
            lambda v: v[0].gelu().sum(),
            [np.array([-1.2, -0.3, 0.4, 1.5])],
        )

    def test_softmax_cross_entropy(self):
        logits = np.array([
            [0.4, -0.2, 0.8],
            [-0.5, 0.9, 0.1],
        ])
        labels = np.array([2, 1])

        self.assert_gradients_match(
            lambda v: v[0].softmax_cross_entropy(labels),
            [logits],
        )
    def test_sum_axis_0(self):
        self.assert_gradients_match(
            lambda v: (v[0].sum(axis=0) * np.array([0.7, -0.2, 1.1])).sum(),
            [np.array([[0.2, -0.4, 0.7],
                       [0.5,  0.8, 1.2]])],
        )

    def test_sum_axis_1(self):
        self.assert_gradients_match(
            lambda v: (v[0].sum(axis=1) * np.array([0.7, -0.2])).sum(),
            [np.array([[0.2, -0.4, 0.7],
                       [0.5,  0.8, 1.2]])],
        )

    def test_mean_axis_0(self):
        self.assert_gradients_match(
            lambda v: (v[0].mean(axis=0) * np.array([0.7, -0.2, 1.1])).sum(),
            [np.array([[0.2, -0.4, 0.7],
                       [0.5,  0.8, 1.2]])],
        )

    def test_mean_axis_1(self):
        self.assert_gradients_match(
            lambda v: (v[0].mean(axis=1) * np.array([0.7, -0.2])).sum(),
            [np.array([[0.2, -0.4, 0.7],
                       [0.5,  0.8, 1.2]])],
        )

    def test_sum_axis_0_keepdims(self):
        self.assert_gradients_match(
            lambda v: (
                v[0].sum(axis=0, keepdims=True)
                * np.array([[0.7, -0.2, 1.1]])
            ).sum(),
            [np.array([[0.2, -0.4, 0.7],
                       [0.5,  0.8, 1.2]])],
        )

    def test_mean_axis_1_keepdims(self):
        self.assert_gradients_match(
            lambda v: (
                v[0].mean(axis=1, keepdims=True)
                * np.array([[0.7], [-0.2]])
            ).sum(),
            [np.array([[0.2, -0.4, 0.7],
                       [0.5,  0.8, 1.2]])],
        )



if __name__ == "__main__":
    unittest.main()
