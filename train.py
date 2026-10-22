"""Train the custom NumPy MLP on a synthetic three-class dataset."""
import numpy as np
from model import MLPClassifier
from masked_parameter import MaskedSGD

def make_dataset(seed=7, points_per_class=250):
    rng = np.random.default_rng(seed)
    centers = np.array([[-2.,-1.],[2.,-.5],[0.,2.]])
    xs, ys = [], []
    for label, center in enumerate(centers):
        xs.append(rng.normal(loc=center, scale=.75, size=(points_per_class,2)))
        ys.append(np.full(points_per_class,label,dtype=np.int64))
    x, y = np.vstack(xs), np.concatenate(ys)
    order = rng.permutation(len(y)); x,y = x[order],y[order]
    split = int(.8*len(y))
    return x[:split],y[:split],x[split:],y[split:]

def main():
    xtr,ytr,xte,yte = make_dataset()
    model = MLPClassifier(2,16,3,seed=42)
    opt = MaskedSGD(model.parameters(),learning_rate=0.05,momentum=0.9)
    rng = np.random.default_rng(42)
    for epoch in range(1,101):
        order = rng.permutation(len(ytr)); losses=[]
        for start in range(0,len(ytr),64):
            ids = order[start:start+64]
            opt.zero_grad()
            loss = model(xtr[ids]).softmax_cross_entropy(ytr[ids])
            loss.backward(); opt.step()
            losses.append(float(loss.data))
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | loss={np.mean(losses):.4f} "
                  f"| train_acc={model.accuracy(xtr,ytr):.3f} "
                  f"| test_acc={model.accuracy(xte,yte):.3f}")
    print(f"\nFinal test accuracy: {model.accuracy(xte,yte):.3f}")
    print("Parameter count:", sum(p.data.size for p in model.parameters()))

if __name__ == "__main__": main()
