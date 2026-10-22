import unittest
import numpy as np
from model import MLPClassifier
from masked_parameter import MaskedSGD
from train import make_dataset

class TestModel(unittest.TestCase):
    def test_model_output_shape(self):
        self.assertEqual(MLPClassifier(2,8,3)(np.zeros((5,2))).shape,(5,3))
    def test_training_reduces_loss(self):
        x,y,_,_=make_dataset(seed=12,points_per_class=80)
        model=MLPClassifier(2,12,3,seed=3); opt=MaskedSGD(model.parameters(),learning_rate=0.05,momentum=0.0)
        initial=float(model(x).softmax_cross_entropy(y).data)
        for _ in range(80):
            opt.zero_grad(); loss=model(x).softmax_cross_entropy(y); loss.backward(); opt.step()
        self.assertLess(float(model(x).softmax_cross_entropy(y).data),initial)
    def test_optimizer_updates_weights(self):
        model=MLPClassifier(2,4,3); before=model.W1.data.copy()
        opt=MaskedSGD(model.parameters(),learning_rate=0.05,momentum=0.0)
        model(np.ones((3,2))).softmax_cross_entropy(np.array([0,1,2])).backward(); opt.step()
        self.assertFalse(np.allclose(before,model.W1.data))

if __name__=="__main__": unittest.main()
