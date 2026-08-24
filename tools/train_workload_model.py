import os

from tools.workload_model import (
    DEFAULT_WEIGHTS_PATH,
    build_model,
    calculate_weighted_score,
)


def generate_synthetic_data(torch, num_samples=500):
    raw = torch.rand(num_samples, 5)
    features = torch.zeros_like(raw)
    labels = torch.zeros(num_samples, dtype=torch.long)

    for index in range(num_samples):
        hours_study = raw[index, 0] * 8
        hours_sport = raw[index, 1] * 10
        hours_work = raw[index, 2] * 20
        deadlines = int(raw[index, 3] * 4)
        sleep_hours = raw[index, 4] * 3 + 5
        score = calculate_weighted_score(
            [hours_study, hours_sport, hours_work, deadlines, sleep_hours]
        )
        label = 0 if score < 5 else 1 if score < 10 else 2 if score < 15 else 3
        features[index] = torch.tensor(
            [hours_study, hours_sport, hours_work, deadlines, sleep_hours]
        )
        labels[index] = label
    return features, labels


def train(epochs=300, output_path=DEFAULT_WEIGHTS_PATH):
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required; install it with pip install -r requirements-ml.txt"
        ) from exc

    torch.manual_seed(7)
    features, labels = generate_synthetic_data(torch)
    model = build_model()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_function(model(features), labels)
        loss.backward()
        optimizer.step()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    train()
