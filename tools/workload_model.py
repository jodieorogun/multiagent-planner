import os
from typing import List


DEFAULT_WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "saved_models",
    "workload_net.pth",
)


class WorkloadModelError(RuntimeError):
    """Raised when the optional neural workload predictor is unavailable."""


def calculate_weighted_score(features: List[float]) -> float:
    hours_study, hours_sport, hours_work, deadlines, sleep_hours = (
        float(value) for value in features
    )
    recovery_credit = max(sleep_hours - 7.0, 0.0)
    return (
        hours_study * 0.7
        + hours_sport * 0.5
        + hours_work * 0.8
        + deadlines
        - recovery_credit
    )


def heuristic_predict(features: List[float]) -> int:
    weighted_score = calculate_weighted_score(features)
    if weighted_score < 5:
        return 0
    if weighted_score < 10:
        return 1
    if weighted_score < 15:
        return 2
    return 3


def build_model():
    try:
        import torch.nn as nn
    except ImportError as exc:
        raise WorkloadModelError(
            "PyTorch is required for the neural workload model; "
            "install requirements-ml.txt"
        ) from exc

    class WorkloadNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(5, 64),
                nn.ReLU(),
                nn.Linear(64, 16),
                nn.ReLU(),
                nn.Linear(16, 16),
                nn.ReLU(),
                nn.Linear(16, 4),
            )

        def forward(self, values):
            return self.layers(values)

    return WorkloadNet()


def neural_predict(features: List[float], weights_path: str = None) -> int:
    try:
        import torch
    except ImportError as exc:
        raise WorkloadModelError(
            "PyTorch is required for the neural workload model; "
            "install requirements-ml.txt"
        ) from exc

    path = weights_path or DEFAULT_WEIGHTS_PATH
    if not os.path.exists(path):
        raise WorkloadModelError(
            f"Workload weights not found at {path}; run tools/train_workload_model.py"
        )

    model = build_model()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    values = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        return int(torch.argmax(model(values), dim=1).item())
