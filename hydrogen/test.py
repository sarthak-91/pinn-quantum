import torch
import torch.nn as nn
import os
from scripts.network import NN
import re


models_dir = "models"  # Change this to your actual directory path

pattern = re.compile(r"state_n_(\d+)_l_(\d+)\.pt")
  # Adjust if your filenames are different
results = []

for filename in os.listdir(models_dir):
    print(filename)
    match = pattern.match(filename)
    print(match)
    if match:
        n, l = int(match.group(1)), int(match.group(2))
        path = os.path.join(models_dir, filename)

        model = NN([50])  # single hidden layer with 50 neurons
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        model.eval()

        # Forward pass with dummy input to compute exponential_coeff
        with torch.no_grad():
            _ = model(torch.tensor([[0.1]]), l=l)  # small dummy input

        coeff = model.exponential_coeff.item()
        results.append(((n, l), coeff))

# === Output the results ===
for (n, l), coeff in sorted(results):
    print(f"n={n}, l={l} -> exponential_coeff = {coeff:.6f}")