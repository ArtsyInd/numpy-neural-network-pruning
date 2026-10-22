import unittest
import numpy as np

from tensor import Tensor
from optim import Adam


class TestAdam(unittest.TestCase):
    def test_updates_parameters_and_keeps_state_finite(self):
        p = Tensor(np.array([1.0, -2.0]), True)
        optimizer = Adam([p], learning_rate=0.01)
        original = p.data.copy()

        optimizer.zero_grad()
        loss = (p * p).mean()
        loss.backward()
        optimizer.step()

        self.assertFalse(np.array_equal(original, p.data))
        self.assertEqual(optimizer.t, 1)
        self.assertTrue(np.all(np.isfinite(p.data)))
        self.assertTrue(np.all(np.isfinite(optimizer.m[0])))
        self.assertTrue(np.all(np.isfinite(optimizer.v[0])))

    def test_multiple_steps(self):
        p = Tensor(np.array([2.0, -1.0]), True)
        optimizer = Adam([p], learning_rate=0.01)

        for _ in range(10):
            optimizer.zero_grad()
            loss = (p * p).mean()
            loss.backward()
            optimizer.step()

        self.assertEqual(optimizer.t, 10)
        self.assertTrue(np.all(np.isfinite(p.data)))
        self.assertLess(np.mean(p.data**2), 2.5)


if __name__ == "__main__":
    unittest.main()
