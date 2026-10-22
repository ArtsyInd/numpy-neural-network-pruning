import unittest
import numpy as np
from tensor import Tensor
from masked_parameter import MaskedParameter, MaskedSGD, MaskedAdam

class TestMaskedParameters(unittest.TestCase):
    def test_forward_and_gradient_mask(self):
        p = MaskedParameter(Tensor(np.array([[2., 3.], [4., 5.]]), True))
        p.set_mask([[True, False], [True, True]])
        y = Tensor(np.ones((1, 2))) @ p.effective_tensor()
        np.testing.assert_allclose(y.data, [[6., 5.]])
        y.sum().backward()
        np.testing.assert_allclose(p.grad, [[1., 0.], [1., 1.]])

    def test_sgd_momentum_prune_revive_resets_state(self):
        p = MaskedParameter(np.array([1., 2.]))
        opt = MaskedSGD([p], learning_rate=.1, momentum=.9)
        opt.zero_grad(); (p.effective_tensor() * np.array([1., 1.])).sum().backward(); opt.step()
        self.assertNotEqual(opt.velocity[0][1], 0.)
        p.set_mask([True, False])
        opt.zero_grad(); (p.effective_tensor() * np.array([1., 100.])).sum().backward()
        self.assertEqual(p.grad[1], 0.)
        opt.step()
        self.assertEqual(p.data[1], 0.); self.assertEqual(opt.velocity[0][1], 0.)
        p.set_mask([True, True])
        opt.zero_grad(); (p.effective_tensor() * np.array([0., 2.])).sum().backward(); opt.step()
        self.assertAlmostEqual(p.data[1], -.2, places=12)
        self.assertAlmostEqual(opt.velocity[0][1], 2., places=12)

    def test_adam_moments_and_step_reset_on_revival(self):
        p = MaskedParameter(np.array([1., 2.]))
        opt = MaskedAdam([p], learning_rate=.01)
        opt.zero_grad(); (p.effective_tensor() * np.array([1., 3.])).sum().backward(); opt.step()
        self.assertGreater(opt.t[0][1], 0); self.assertNotEqual(opt.m[0][1], 0.)
        p.set_mask([True, False])
        opt.zero_grad(); (p.effective_tensor() * np.array([0., 99.])).sum().backward(); opt.step()
        self.assertEqual(p.data[1], 0.); self.assertEqual(opt.m[0][1], 0.); self.assertEqual(opt.v[0][1], 0.)
        self.assertEqual(opt.t[0][1], 0)
        p.set_mask([True, True])
        opt.zero_grad(); (p.effective_tensor() * np.array([0., 2.])).sum().backward(); opt.step()
        self.assertEqual(opt.t[0][1], 1)
        self.assertLess(p.data[1], 0.)

    def test_multiple_mask_changes_between_optimizer_steps_are_not_missed(self):
        p = MaskedParameter(np.array([1.]))
        opt = MaskedSGD([p], learning_rate=.1, momentum=.9)
        opt.zero_grad(); (p.effective_tensor() * 3.).sum().backward(); opt.step()
        p.set_mask([False]); p.set_mask([True])  # toggle twice before step
        opt.zero_grad(); (p.effective_tensor() * 2.).sum().backward(); opt.step()
        self.assertAlmostEqual(opt.velocity[0][0], 2.)

    def test_mask_shape_validation(self):
        p = MaskedParameter(np.ones((2, 2)))
        with self.assertRaises(ValueError): p.set_mask(np.ones(4, dtype=bool))

if __name__ == '__main__': unittest.main()
