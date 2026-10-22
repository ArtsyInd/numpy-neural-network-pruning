import unittest
import numpy as np
from tensor import Tensor
from pruning import MaskedParameter, MaskedAdam


class TestMaskingAndRevival(unittest.TestCase):
    def test_masked_connection_has_zero_forward_contribution(self):
        # One input, two output connections. The second is masked.
        x = Tensor(np.array([[3.0]]))
        w = MaskedParameter(Tensor(np.array([[2.0, 7.0]]), True),
                            mask=np.array([[1.0, 0.0]]))
        y = x @ w.effective()
        np.testing.assert_allclose(y.data, np.array([[6.0, 0.0]]))

    def test_masked_connection_has_zero_gradient(self):
        x = Tensor(np.array([[3.0]]))
        w = MaskedParameter(Tensor(np.array([[2.0, 7.0]]), True),
                            mask=np.array([[1.0, 0.0]]))
        w.effective().sum()  # deliberately ensure graph can be built repeatedly
        loss = (x @ w.effective()).sum()
        loss.backward()
        np.testing.assert_allclose(w.value.grad, np.array([[3.0, 0.0]]))

    def test_pruned_weight_stays_zero_after_optimizer_step(self):
        p = MaskedParameter(Tensor(np.array([[1.0, 2.0]]), True))
        opt = MaskedAdam([p], learning_rate=0.1)
        # Seed nonzero optimizer state by taking an update.
        opt.zero_grad()
        p.effective().sum().backward()
        opt.step()
        opt.set_mask(p, np.array([[1.0, 0.0]]))
        self.assertEqual(p.value.data[0, 1], 0.0)

        opt.zero_grad()
        p.effective().sum().backward()
        opt.step()
        self.assertEqual(p.value.data[0, 1], 0.0)
        self.assertEqual(p.value.grad[0, 1], 0.0)
        self.assertEqual(opt.m[id(p)][0, 1], 0.0)
        self.assertEqual(opt.v[id(p)][0, 1], 0.0)

    def test_revival_resets_stale_adam_state(self):
        p = MaskedParameter(Tensor(np.array([[1.0]]), True))
        opt = MaskedAdam([p], learning_rate=0.01)

        # Create momentum in the active connection.
        for _ in range(4):
            opt.zero_grad()
            (p.effective() * 2.0).sum().backward()
            opt.step()
        key = id(p)
        self.assertNotEqual(opt.m[key][0, 0], 0.0)
        self.assertNotEqual(opt.v[key][0, 0], 0.0)
        self.assertEqual(opt.age[key][0, 0], 4)

        # Prune it: weight and per-parameter moments must be zero.
        opt.set_mask(p, np.array([[0.0]]))
        self.assertEqual(p.value.data[0, 0], 0.0)
        self.assertEqual(opt.m[key][0, 0], 0.0)
        self.assertEqual(opt.v[key][0, 0], 0.0)
        self.assertEqual(opt.age[key][0, 0], 0)

        # While pruned, optimizer steps cannot change weight or accumulate state.
        for _ in range(3):
            opt.zero_grad()
            (p.effective() * 10.0).sum().backward()
            opt.step()
        self.assertEqual(p.value.data[0, 0], 0.0)
        self.assertEqual(opt.m[key][0, 0], 0.0)
        self.assertEqual(opt.v[key][0, 0], 0.0)
        self.assertEqual(opt.age[key][0, 0], 0)

        # Revive it: state remains clean; the next gradient starts fresh.
        opt.set_mask(p, np.array([[1.0]]))
        self.assertEqual(opt.m[key][0, 0], 0.0)
        self.assertEqual(opt.v[key][0, 0], 0.0)
        self.assertEqual(opt.age[key][0, 0], 0)
        before = p.value.data.copy()
        opt.zero_grad()
        (p.effective() * 3.0).sum().backward()
        self.assertAlmostEqual(p.value.grad[0, 0], 3.0)
        opt.step()
        self.assertLess(p.value.data[0, 0], before[0, 0])
        self.assertTrue(np.isfinite(p.value.data[0, 0]))
        self.assertTrue(np.isfinite(opt.m[key][0, 0]))
        self.assertTrue(np.isfinite(opt.v[key][0, 0]))
        self.assertEqual(opt.age[key][0, 0], 1)

    def test_mask_shape_and_binary_validation(self):
        p = MaskedParameter(Tensor(np.ones((2, 2)), True))
        opt = MaskedAdam([p])
        with self.assertRaises(ValueError):
            opt.set_mask(p, np.ones((3, 3)))
        with self.assertRaises(ValueError):
            opt.set_mask(p, np.array([[1.0, 0.5], [1.0, 1.0]]))


if __name__ == "__main__":
    unittest.main()
