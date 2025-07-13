import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

# Generate x coordinates from 0 to 2*pi with 0.1 step
x = np.arange(0, 10 * np.pi, 0.1)
# Compute y coordinates using sine
y = np.sin(x)


# Set seaborn style
sns.set_theme(style="darkgrid")

# Create the plot
plt.figure(figsize=(8, 4))
plt.plot(x, y, label='sin(x)')
plt.title('Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.tight_layout()
plt.show()

# Save x and y coordinates to a CSV file
df = pd.DataFrame({'x': x, 'y': y})
df.to_csv('sine_coordinates_test.csv', index=False)
