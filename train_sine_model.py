import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Load the data
df = pd.read_csv('sine_coordinates.csv')

# Prepare input and output tensors
X = torch.tensor(df['x'].values, dtype=torch.float32).unsqueeze(1)  # shape (N, 1)
y = torch.tensor(df['y'].values, dtype=torch.float32).unsqueeze(1)  # shape (N, 1)

# Define a simple neural network
class SineNet(nn.Module):
    def __init__(self):
        super(SineNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
    def forward(self, x):
        return self.net(x)

model = SineNet()

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100_000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

# Save the trained model
torch.save(model.state_dict(), 'sine_model.pth')
print('Training complete. Model saved as sine_model.pth')
