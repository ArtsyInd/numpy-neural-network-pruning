import unittest
import numpy as np
from tensor import Tensor

def finite_difference(fn, x, eps=1e-6):
    g = np.zeros_like(x, dtype=float)
    for idx in np.ndindex(x.shape):
        old=x[idx]; x[idx]=old+eps; plus=float(fn(x).data)
        x[idx]=old-eps; minus=float(fn(x).data); x[idx]=old
        g[idx]=(plus-minus)/(2*eps)
    return g

class TestAutodiff(unittest.TestCase):
    def check(self, fn, values, atol=2e-5):
        values=np.asarray(values,dtype=float)
        t=Tensor(values.copy(),True); fn(t).backward()
        numeric=finite_difference(lambda a: fn(Tensor(a)),values.copy())
        np.testing.assert_allclose(t.grad,numeric,atol=atol,rtol=atol)
    def test_add(self): self.check(lambda x:(x+1.7).sum(),[.2,-.4,1.1])
    def test_sub(self): self.check(lambda x:(x-1.7).sum(),[.2,-.4,1.1])
    def test_mul(self): self.check(lambda x:(x*x).sum(),[.2,-.4,1.1])
    def test_div(self): self.check(lambda x:(x/2.3).sum(),[.2,.4,1.1])
    def test_reverse_div(self): self.check(lambda x:(2/x).sum(),[.7,1.2,2.])
    def test_matmul(self):
        a=np.array([[.2,.3],[-.1,.5]]); b=np.array([[.4,-.2],[.7,.1]])
        ta,tb=Tensor(a,True),Tensor(b,True); (ta@tb).sum().backward()
        np.testing.assert_allclose(ta.grad,np.ones((2,2))@b.T)
        np.testing.assert_allclose(tb.grad,a.T@np.ones((2,2)))
        numeric_a=finite_difference(lambda z:(Tensor(z)@Tensor(b)).sum(),a.copy())
        numeric_b=finite_difference(lambda z:(Tensor(a)@Tensor(z)).sum(),b.copy())
        np.testing.assert_allclose(ta.grad,numeric_a,atol=2e-5,rtol=2e-5)
        np.testing.assert_allclose(tb.grad,numeric_b,atol=2e-5,rtol=2e-5)
    def test_sum(self): self.check(lambda x:x.sum(),[[.2,.3],[.4,.5]])
    def test_mean(self): self.check(lambda x:x.mean(),[[.2,.3],[.4,.5]])
    def test_mean_axis(self):
        x=np.array([[.2,.3],[.4,.5]]); t=Tensor(x,True)
        t.mean(axis=0).sum().backward()
        np.testing.assert_allclose(t.grad,np.full_like(x,.5))
    def test_relu(self): self.check(lambda x:x.relu().sum(),[-1.2,.4,2.])
    def test_tanh(self): self.check(lambda x:x.tanh().sum(),[-.7,.2,1.1])
    def test_exp(self): self.check(lambda x:x.exp().sum(),[-.7,.2,.8])
    def test_log(self): self.check(lambda x:x.log().sum(),[.7,1.2,2.])
    def test_cross_entropy(self):
        values=np.array([[1.2,-.3,.7],[.1,.8,-.4]]); labels=np.array([0,2])
        self.check(lambda x:x.softmax_cross_entropy(labels),values)
    def test_broadcast_gradient(self):
        x=Tensor(np.arange(6.).reshape(2,3),True); b=Tensor([.1,.2,.3],True)
        (x+b).sum().backward()
        np.testing.assert_allclose(x.grad,np.ones((2,3)))
        np.testing.assert_allclose(b.grad,[2.,2.,2.])
    def test_shared_node_gradient_accumulation(self):
        x=Tensor([2.,3.],True); (x*x+x).sum().backward()
        np.testing.assert_allclose(x.grad,[5.,7.])
    def test_non_scalar_backward_requires_gradient(self):
        with self.assertRaises(ValueError): Tensor([1.,2.],True).backward()
    def test_large_logits_stable(self):
        t=Tensor([[10000.,9999.,-10000.]],True)
        loss=t.softmax_cross_entropy([0]); self.assertTrue(np.isfinite(loss.data))
        loss.backward(); self.assertTrue(np.all(np.isfinite(t.grad)))

if __name__=="__main__": unittest.main()
