import os
import torch
import torch.nn as nn
from typing import List

class WorkloadNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def loadWorkloadModel(weightsPath: str = None) -> WorkloadNet:
    if weightsPath is None:
        weightsPath = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "saved_models",
            "workloadNet.pth",
        )

    model = WorkloadNet()
    stateDict = torch.load(weightsPath, map_location="cpu")
    model.load_state_dict(stateDict)
    model.eval()
    return model


def predictWorkload(features: List[float]) -> int:
    """
    features = [hoursStudy, hoursSport, hoursWork, numDeadlines, sleepHours]
    returns int in {0,1,2,3}
    """
    model = loadWorkloadModel()
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        prediction = torch.argmax(logits, dim=1).item()
    return prediction
