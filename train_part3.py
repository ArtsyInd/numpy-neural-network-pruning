"""Part 3: train an MLP while gradually pruning connections."""
import argparse, csv
import numpy as np
import matplotlib.pyplot as plt
from tensor import Tensor
from layers import Linear, ReLU, Sequential
from optim import Adam
from masked_parameter import MaskedAdam
from pruning import GlobalPruner, cubic_sparsity
from train_part2 import make_spiral_data, evaluate

def make_prunable_model(seed=123):
    rng = np.random.default_rng(seed)
    return Sequential(
        Linear(2, 64, rng, initialization="he", prunable=True), ReLU(),
        Linear(64, 64, rng, initialization="he", prunable=True), ReLU(),
        Linear(64, 3, rng, initialization="xavier", prunable=True),
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--target-sparsity", type=float, default=0.80)
    parser.add_argument("--prune-method", choices=["taylor", "magnitude"], default="taylor")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--prune-start-fraction", type=float, default=0.20)
    parser.add_argument("--prune-end-fraction", type=float, default=0.80)
    parser.add_argument("--prune-every", type=int, default=5)
    parser.add_argument("--output-prefix", default="part3")
    a = parser.parse_args()
    if a.epochs < 2 or a.batch_size < 1 or a.prune_every < 1: raise ValueError("Invalid epochs/batch/prune interval")
    if not 0 <= a.target_sparsity < 1: raise ValueError("target-sparsity must be in [0,1)")
    if not 0 <= a.prune_start_fraction < a.prune_end_fraction <= 1: raise ValueError("Require 0 <= start < end <= 1")

    X, y = make_spiral_data(seed=42)
    split_rng = np.random.default_rng(99); idx = split_rng.permutation(len(y)); split = int(.8*len(y))
    tr, te = idx[:split], idx[split:]
    X_train, y_train, X_test, y_test = X[tr], y[tr], X[te], y[te]
    mean, std = X_train.mean(axis=0), X_train.std(axis=0); std[std == 0] = 1.0
    X_train, X_test = (X_train-mean)/std, (X_test-mean)/std

    model = make_prunable_model(a.seed)
    masked_params = model.masked_parameters()
    pruner = GlobalPruner(masked_params, method=a.prune_method)
    masked_opt = MaskedAdam(masked_params, learning_rate=a.learning_rate)
    dense_opt = Adam(model.dense_parameters(), learning_rate=a.learning_rate)
    start = int(a.epochs*a.prune_start_fraction)
    end = max(start+1, int(a.epochs*a.prune_end_fraction))
    rng = np.random.default_rng(a.seed+1000); history=[]
    initial_loss, initial_acc = evaluate(model, X_test, y_test)
    print(f"Initial test loss={initial_loss:.4f}, accuracy={initial_acc:.3f}")

    for epoch in range(1, a.epochs+1):
        order = rng.permutation(len(y_train))
        for pos in range(0, len(order), a.batch_size):
            ids = order[pos:pos+a.batch_size]
            masked_opt.zero_grad(); dense_opt.zero_grad()
            loss = model(Tensor(X_train[ids])).softmax_cross_entropy(y_train[ids])
            if not np.isfinite(float(np.asarray(loss.data))): raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            if start < end and start <= epoch <= end and (epoch-start) % a.prune_every == 0:
                target = cubic_sparsity(epoch, start, end, a.target_sparsity)
                pruner.prune_to_sparsity(target)  # before step: current Taylor gradients are available
            masked_opt.step(); dense_opt.step()

        train_loss, train_acc = evaluate(model, X_train, y_train)
        test_loss, test_acc = evaluate(model, X_test, y_test)
        st = pruner.statistics()
        history.append({"epoch":epoch,"train_loss":train_loss,"train_accuracy":train_acc,
          "test_loss":test_loss,"test_accuracy":test_acc,"method":a.prune_method,"seed":a.seed,
          "target_sparsity":a.target_sparsity,"active_connections":st["active_connections"],
          "total_connections":st["total_connections"],"achieved_sparsity":st["sparsity"]})
        if epoch == 1 or epoch % 10 == 0 or epoch == a.epochs:
            print(f"Epoch {epoch:3d}/{a.epochs} | train acc={train_acc:.3f} test acc={test_acc:.3f} sparsity={st['sparsity']:.3f}")

    csv_path=f"{a.output_prefix}_metrics.csv"
    with open(csv_path,"w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)
    ep=[r["epoch"] for r in history]
    plt.figure(); plt.plot(ep,[r["train_accuracy"] for r in history],label="Train accuracy"); plt.plot(ep,[r["test_accuracy"] for r in history],label="Test accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title(f"Part 3: {a.prune_method}, target sparsity {a.target_sparsity:.0%}"); plt.legend(); plt.tight_layout()
    plot_path=f"{a.output_prefix}_accuracy.png"; plt.savefig(plot_path,dpi=150); plt.close()
    print(f"Saved {csv_path} and {plot_path}")
    print(f"Final test accuracy={history[-1]['test_accuracy']:.4f}; achieved sparsity={history[-1]['achieved_sparsity']:.4f}")

if __name__ == "__main__": main()
