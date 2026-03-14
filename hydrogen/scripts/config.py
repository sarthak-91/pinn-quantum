import os
import torch 
PROJECT_ROOT = '/Users/sarthakbhattarai/Documents/code/python/pinn/hydrogen'
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models')
PLOT_PATH = os.path.join(PROJECT_ROOT,'plots')
CONVERGE_PATH = os.path.join(PROJECT_ROOT,'convergence')
WF_PATH = os.path.join(PROJECT_ROOT,'wavefunctions')
CSV_FILE = os.path.join(PROJECT_ROOT,'registry.csv')
LOGGING_PATH = os.path.join(PROJECT_ROOT,'logs')

device = "cpu"
#device = "mps"
print("Using",device)

torch.manual_seed(42)