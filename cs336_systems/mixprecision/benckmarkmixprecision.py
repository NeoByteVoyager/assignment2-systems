import torch
import torch.nn as nn
import torch.optim as optim


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        print("after fc1, dtype:", x.dtype)
        x = self.ln(x)
        print("after layernorm, dtype:", x.dtype)
        x = self.fc2(x)
        print("after fc2, dtype:", x.dtype)

        return x


device = "cuda" if torch.cuda.is_available() else "cpu"
model = ToyModel(10, 1).to(device)
criterion = nn.MSELoss()

inputs = torch.randn(4, 10, device=device)
targets = torch.randn(4, 1, device=device)

optimizer = optim.SGD(model.parameters(), lr=0.01)



with torch.autocast(device, dtype=torch.float16):
    print("model weights dtype:", model.fc1.weight.dtype)
    outputs = model(inputs)

    optimizer.zero_grad()
    loss = criterion(outputs, targets)
    loss.backward()
    print("loss dtype:", loss.dtype)
    print("model weights grad dtype:",model.fc1.weight.grad.dtype)

    optimizer.step()


