import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
csv_path = 'sine_coordinates.csv'
df = pd.read_csv(csv_path)
X = torch.tensor(df['x'].values, dtype=torch.float32).unsqueeze(1)

# Define the same model architecture
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

# Load the trained model
model = SineNet()
print(model.state_dict().keys())  # Print model state_dict keys for debugging
model.load_state_dict(torch.load('sine_model.pth'))
model.eval()

# Make predictions
with torch.no_grad():
    y_pred = model(X).squeeze().numpy()

# Plot the predictions vs true values
sns.set_theme(style="darkgrid")
plt.figure(figsize=(8, 4))
plt.plot(df['x'], df['y'], label='True sin(x)', color='blue')
plt.plot(df['x'], y_pred, label='Predicted', color='red', linestyle='--')
plt.title('Sine Model Predictions')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.tight_layout()
plt.show()
