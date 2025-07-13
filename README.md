# Sine Wave Neural Network Demo

This repository demonstrates generating, visualizing, and modeling a sine wave using Python, NumPy, Matplotlib, Seaborn, Pandas, and PyTorch.

## Features
- Generate x and y coordinates for a sine wave and save them as a CSV file
- Visualize the sine wave using Matplotlib and Seaborn
- Train a simple neural network with PyTorch to learn the sine function
- Predict and plot the neural network's output against the true sine values

## Files
- `sine.py`: Generates sine wave data, visualizes it, and saves it to `sine_coordinates.csv`.
- `train_sine_model.py`: Loads the CSV data, trains a neural network to fit the sine function, and saves the model as `sine_model.pth`.
- `predict_and_plot.py`: Loads the trained model, predicts sine values for the x coordinates, and plots both the true and predicted values.
- `sine_coordinates.csv`: The generated dataset of x and y values.

## Requirements
- Python 3.7+
- numpy
- pandas
- matplotlib
- seaborn
- torch (PyTorch)

Install dependencies with:
```bash
pip install numpy pandas matplotlib seaborn torch
```

## Usage
1. Generate and visualize sine data:
    ```bash
    python sine.py
    ```
2. Train the neural network:
    ```bash
    python train_sine_model.py
    ```
3. Predict and plot results:
    ```bash
    python predict_and_plot.py
    ```

## License
MIT License
