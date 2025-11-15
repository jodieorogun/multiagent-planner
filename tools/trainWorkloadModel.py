import os
import torch
import torch.nn as nn
import torch.optim as optim

class WorkloadNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 16) # 5 input features, 16 hidden units
        self.fc2 = nn.Linear(16, 4) # 4 output classes, representing workload levels

    def forward(self, x):
        x = torch.relu(self.fc1(x)) # hidden layer with ReLU activation
        x = self.fc2(x)
        return x


def generateSyntheticData(num_samples: int = 400):
    X = torch.rand(num_samples, 5) # creates 400 x 5 tensor with random values between 0 and 1
    y = torch.zeros(num_samples, dtype=torch.long)

    for i in range(num_samples):
        hours_study = X[i, 0] * 8
        hours_sport = X[i, 1] * 10
        hours_work = X[i, 2] * 20
        num_deadlines = int(X[i, 3] * 4)
        sleep_hours = X[i, 4] * 3 + 5  

        score = ( # weighted sum to determine workload level
            hours_study * 0.7
            + hours_sport * 0.5
            + hours_work * 0.8
            + num_deadlines * 1.0
            - sleep_hours * 1.0
        )

        if score < 5:
            label = 0 # light workload
        elif score < 10:
            label = 1 # moderate workload
        elif score < 15:
            label = 2 # high workload
        else:
            label = 3  # burnout risk

        X[i] = torch.tensor(
            [hours_study, hours_sport, hours_work, float(num_deadlines), sleep_hours],
            dtype=torch.float32,
        ) # overwrite with scaled values 
        y[i] = label

    return X, y


def trainWorkloadModel():
    X, y = generateSyntheticData()

    model = WorkloadNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(80): 
        optimizer.zero_grad() # clear gradients from previous step
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/80 - Loss: {loss.item():.4f}")

    os.makedirs("saved_models", exist_ok=True)
    save_path = os.path.join("saved_models", "workloadNet.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")


if __name__ == "__main__":
    trainWorkloadModel()
