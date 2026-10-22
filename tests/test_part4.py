import unittest
import numpy as np

from layers import Linear, ReLU, Sequential
from sparse_inference import predict_sparse, dense_zero_predict


class TestPart4InferenceCost(unittest.TestCase):
    def test_sparse_aware_inference_matches_dense_zero_filled(self):
        rng = np.random.default_rng(2026)
        model = Sequential(
            Linear(4, 6, rng, prunable=True),
            ReLU(),
            Linear(6, 3, rng, initialization="xavier", prunable=True),
        )
        masks = [
            rng.random((4, 6)) > 0.5,
            rng.random((6, 3)) > 0.4,
        ]
        for parameter, mask in zip(model.masked_parameters(), masks):
            parameter.set_mask(mask, reset_values=True)
        x = rng.normal(size=(9, 4))
        sparse_logits, active_multiplications = predict_sparse(model, x)
        dense_logits = dense_zero_predict(model, x)
        np.testing.assert_allclose(sparse_logits, dense_logits, rtol=1e-9, atol=1e-10)
        expected = len(x) * sum(int(mask.sum()) for mask in masks)
        self.assertEqual(active_multiplications, expected)

    def test_all_active_multiplication_count_matches_dense_count(self):
        rng = np.random.default_rng(7)
        model = Sequential(
            Linear(3, 4, rng, prunable=True),
            ReLU(),
            Linear(4, 2, rng, initialization="xavier", prunable=True),
        )
        x = rng.normal(size=(5, 3))
        _, active_multiplications = predict_sparse(model, x)
        dense_count = len(x) * sum(
            layer.weight.data.size for layer in model.layers
            if isinstance(layer, Linear)
        )
        self.assertEqual(active_multiplications, dense_count)


if __name__ == "__main__":
    unittest.main()
