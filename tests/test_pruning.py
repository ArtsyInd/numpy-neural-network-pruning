import unittest
import numpy as np
from tensor import Tensor
from layers import Linear, ReLU, Sequential
from masked_parameter import MaskedParameter, MaskedAdam
from pruning import GlobalPruner, cubic_sparsity, taylor_scores
from sparse_inference import sparse_linear, predict_sparse, dense_masked_predict

class TestPruning(unittest.TestCase):
    def test_cubic_schedule(self):
        values=[cubic_sparsity(i,10,30,.8) for i in range(41)]
        self.assertEqual(values[0],0); self.assertAlmostEqual(values[10],0); self.assertAlmostEqual(values[30],.8)
        self.assertTrue(all(x<=y for x,y in zip(values,values[1:])))
    def test_exact_global_sparsity_and_zero_weights(self):
        p1=MaskedParameter(np.arange(1.,7.).reshape(2,3)); p2=MaskedParameter(np.arange(1.,5.).reshape(2,2))
        stats=GlobalPruner([p1,p2],"magnitude").prune_to_sparsity(.4)
        self.assertEqual(stats["pruned_connections"],4); self.assertAlmostEqual(stats["sparsity"],.4)
        for p in (p1,p2): np.testing.assert_array_equal(p.data[~p.mask],0.)
    def test_taylor_scores(self):
        p=MaskedParameter(Tensor([2.,3.,4.],True)); p.grad[:]=[5.,1.,-2.]
        np.testing.assert_allclose(taylor_scores(p),[10.,3.,8.])
    def test_optimizer_state_resets_on_prune(self):
        p=MaskedParameter(np.array([1.,2.])); opt=MaskedAdam([p],learning_rate=.01)
        opt.zero_grad(); (p.effective_tensor()*np.array([1.,2.])).sum().backward(); opt.step()
        self.assertGreater(opt.t[0][1],0); p.set_mask([True,False])
        opt.zero_grad(); (p.effective_tensor()*np.array([1.,2.])).sum().backward(); opt.step()
        self.assertEqual(opt.t[0][1],0); self.assertEqual(p.data[1],0.)
    def test_sparse_linear_equivalence(self):
        rng=np.random.default_rng(4); x=rng.normal(size=(7,5)); w=rng.normal(size=(5,4)); m=rng.random((5,4))>.45; b=rng.normal(size=4)
        out,count=sparse_linear(x,w,m,b); np.testing.assert_allclose(out,x@(w*m)+b,rtol=1e-12,atol=1e-12)
        self.assertEqual(count,int(m.sum())*len(x))
    def test_sparse_model_equivalence(self):
        rng=np.random.default_rng(10); model=Sequential(Linear(2,5,rng,prunable=True),ReLU(),Linear(5,3,rng,initialization="xavier",prunable=True))
        model.masked_parameters()[0].set_mask(np.array([[1,0,1,1,0],[1,1,0,1,1]],bool))
        model.masked_parameters()[1].set_mask(np.array([[1,0,1],[0,1,1],[1,1,0],[1,0,1],[0,1,1]],bool))
        x=rng.normal(size=(8,2)); sparse,_=predict_sparse(model,x)
        np.testing.assert_allclose(sparse,dense_masked_predict(model,x),rtol=1e-12,atol=1e-12)

if __name__ == "__main__": unittest.main()
