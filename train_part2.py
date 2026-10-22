import csv
import numpy as np
import matplotlib.pyplot as plt

from tensor import Tensor
from layers import Linear, ReLU, Sequential
from optim import Adam


def make_spiral_data(n_per_class=300, classes=3, seed=42):
    """Generate a synthetic multi-class spiral dataset."""
    rng = np.random.default_rng(seed)

    X = np.zeros((n_per_class * classes, 2), dtype=np.float64)
    y = np.zeros(n_per_class * classes, dtype=np.int64)

    for class_id in range(classes):
        start = class_id * n_per_class
        end = start + n_per_class

        radius = np.linspace(0.02, 1.0, n_per_class)
        angle = (
            np.linspace(0, 4 * np.pi, n_per_class)
            + class_id * (2 * np.pi / classes)
            + rng.normal(0.0, 0.15, n_per_class)
        )

        X[start:end, 0] = radius * np.cos(angle)
        X[start:end, 1] = radius * np.sin(angle)
        y[start:end] = class_id

    # Shuffle the examples.
    order = rng.permutation(len(y))
    return X[order], y[order]


def make_model(seed=123):
    rng = np.random.default_rng(seed)

    return Sequential(
        Linear(2, 64, rng, initialization="he"),
        ReLU(),
        Linear(64, 64, rng, initialization="he"),
        ReLU(),
        Linear(64, 3, rng, initialization="xavier"),
    )


def evaluate(model, X, y):
    logits = model(Tensor(X))
    loss = logits.softmax_cross_entropy(y)
    predictions = np.argmax(logits.data, axis=1)
    accuracy = np.mean(predictions == y)

    loss_value = float(np.asarray(loss.data))

    if not np.isfinite(loss_value):
        raise FloatingPointError("Non-finite evaluation loss")
    if not np.all(np.isfinite(logits.data)):
        raise FloatingPointError("Non-finite model logits")

    return loss_value, float(accuracy)


def main():
    rng = np.random.default_rng(7)

    X, y = make_spiral_data()

    # Split into train and test sets.
    split_rng = np.random.default_rng(99)
    indices = split_rng.permutation(len(y))
    split = int(0.8 * len(y))

    train_idx = indices[:split]
    test_idx = indices[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Standardize using training data only.
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    model = make_model()
    optimizer = Adam(
        model.parameters(),
        learning_rate=0.003,
    )

    epochs = 500
    batch_size = 64
    history = []

    initial_loss, initial_accuracy = evaluate(
        model, X_test, y_test
    )
    print(
        f"Initial test loss: {initial_loss:.4f}, "
        f"accuracy: {initial_accuracy:.3f}"
    )

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(y_train))

        for start in range(0, len(order), batch_size):
            batch_idx = order[start:start + batch_size]
            xb = X_train[batch_idx]
            yb = y_train[batch_idx]

            optimizer.zero_grad()

            logits = model(Tensor(xb))
            loss = logits.softmax_cross_entropy(yb)

            loss_value = float(np.asarray(loss.data))
            if not np.isfinite(loss_value):
                raise FloatingPointError(
                    f"Non-finite training loss at epoch {epoch}"
                )

            loss.backward()
            optimizer.step()

        train_loss, train_accuracy = evaluate(
            model, X_train, y_train
        )
        test_loss, test_accuracy = evaluate(
            model, X_test, y_test
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        })

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train loss {train_loss:.4f} | "
                f"train acc {train_accuracy:.3f} | "
                f"test loss {test_loss:.4f} | "
                f"test acc {test_accuracy:.3f}"
            )

    # Save the learning-curve data.
    with open(
        "part2_metrics.csv", "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    # Plot loss.
    epochs_axis = [row["epoch"] for row in history]

    plt.figure()
    plt.plot(epochs_axis, [r["train_loss"] for r in history],
             label="Train loss")
    plt.plot(epochs_axis, [r["test_loss"] for r in history],
             label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Part 2: Loss over epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig("part2_loss.png", dpi=150)
    plt.close()

    # Plot accuracy.
    plt.figure()
    plt.plot(epochs_axis, [r["train_accuracy"] for r in history],
             label="Train accuracy")
    plt.plot(epochs_axis, [r["test_accuracy"] for r in history],
             label="Test accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Part 2: Accuracy over epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig("part2_accuracy.png", dpi=150)
    plt.close()

    print("\nTraining complete.")
    print("Saved: part2_metrics.csv, part2_loss.png, part2_accuracy.png")
    print(
        f"Final test loss: {history[-1]['test_loss']:.4f}; "
        f"final test accuracy: {history[-1]['test_accuracy']:.3f}"
    )


if __name__ == "__main__":
    main()