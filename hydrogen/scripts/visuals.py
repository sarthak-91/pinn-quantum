import matplotlib.pyplot as plt
import numpy as np 
import os 
import pandas as pd
import torch 
from scripts.config import * 
def plot_loss_curve(losses, save_path = "training.png"):
    plt.figure(figsize=(10, 6))

    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid()
    plt.savefig(save_path)

def plot_energy_curve(energies, n=1,l=0,path = CONVERGE_PATH):
    plt.figure(figsize=(10, 6))

    plt.plot(energies[500:])
    plt.xlabel("Epoch")
    plt.ylabel("Energy")
    plt.title(f"Energy Convergence, n={n}, l={l}")
    plt.grid()
    plt.savefig(os.path.join(path,f"converge_n_{n}_l_{l}.png"))


def plot_all_wavefunctions(r, model_class,hidden_layers=[10,10], plot_path=PLOT_PATH, model_path=MODEL_PATH, csv_file=CSV_FILE):
    
    def psi_10(x): return x * 2 * np.exp(-x)
    def psi_21(x): return x * (1 / (np.sqrt(6) * 2)) * x * np.exp(-x / 2)
    def psi_20(x): return x * (1 / np.sqrt(2)) * (1 - x / 2) * np.exp(-x / 2)
    def psi_32(x): return x * (4 / (9 * np.sqrt(30))) * (x / 3)**2 * np.exp(-x / 3)
    def psi_30(x): return x * (2 / (3 * np.sqrt(3))) * (1 - (2 * x / 3) + (2 * x**2 / 27)) * np.exp(-x / 3)
    def psi_31(x): return x * (8 / (27 * np.sqrt(6))) * (1 - x / 6) * x * np.exp(-x / 3)
    def psi_40(x): return x * (1 / 4) * (1 - 12 * x / 16 + 32 * x**2 / 256 - 64 / 3 * x**3 / (16**3)) * np.exp(-x / 4)
    def psi_41(r): return r * (64 * np.sqrt(15) / (3 * 16**1.5)) * (r/16 - 4*(r/16)**2 + (16/5)*(r/16)**3) * np.exp(-r/4)

    analytical_map = {
        (1, 0): psi_10,
        (2, 1): psi_21,
        (2, 0): psi_20,
        (3, 2): psi_32,
        (3, 0): psi_30,
        (3, 1): psi_31,
        (4, 0): psi_40,
        (4, 1): psi_41
    }

    if r.ndim == 1:
        r = r.unsqueeze(-1)
    r_np = r.squeeze().cpu().detach().numpy()

    df = pd.read_csv(csv_file)

    for _, row in df.iterrows():
        n = int(row['n'])
        l = int(row['l'])
        energy = float(row['energy'])
        param_file = row['param_file']
        model_file = os.path.join(model_path, param_file)

        if not os.path.exists(model_file):
            print(f"Model file not found: {model_file}")
            continue


        model = model_class(hidden_layers)
        state_dict = torch.load(model_file, map_location=r.device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(r.device)
        model.eval()

        with torch.no_grad():
            psi = model(r,l).squeeze().cpu().numpy()
        if psi[0] < 0: psi = -psi
        plt.figure(figsize=(6, 6))
        plt.plot(r_np, psi, label=f"NN $\\psi_{{{n}{l}}}^2$, E={energy:.5f}", color='blue')

        if (n, l) in analytical_map:
            psi_analytical = analytical_map[(n, l)](r_np)
            plt.plot(r_np, psi_analytical, '--', label=fr"Exact $\psi_{{{n}{l}}}^2$", color='orange')

        plt.title(f"Wavefunction: n={n}, l={l}")
        plt.xlabel("r")
        plt.ylabel(r"$\psi_{nl}^2(r)$")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        dir_path = os.path.join(plot_path, f"n{n}")
        os.makedirs(dir_path, exist_ok=True)

        save_path = os.path.join(dir_path, f"wavefunction_n{n}_l{l}.png")
        plt.savefig(save_path)
        plt.close()


import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_all_paper(r, model_class, hidden_layers=[10, 10],
                           plot_path='plots', model_path='models', csv_file='registry.csv'):
    # Analytical wavefunctions
    def psi_10(x): return x * 2 * np.exp(-x)
    def psi_21(x): return x * (1 / (np.sqrt(6) * 2)) * x * np.exp(-x / 2)
    def psi_20(x): return x * (1 / np.sqrt(2)) * (1 - x / 2) * np.exp(-x / 2)
    def psi_32(x): return x * (4 / (9 * np.sqrt(30))) * (x / 3)**2 * np.exp(-x / 3)
    def psi_30(x): return x * (2 / (3 * np.sqrt(3))) * (1 - (2 * x / 3) + (2 * x**2 / 27)) * np.exp(-x / 3)
    def psi_31(x): return x * (8 / (27 * np.sqrt(6))) * (1 - x / 6) * x * np.exp(-x / 3)
    def psi_40(x): return x * (1 / 4) * (1 - 12 * x / 16 + 32 * x**2 / 256 - 64 / 3 * x**3 / (16**3)) * np.exp(-x / 4)
    def psi_41(r): return r * (64 * np.sqrt(15) / (3 * 16**1.5)) * (r/16 - 4*(r/16)**2 + (16/5)*(r/16)**3) * np.exp(-r/4)

    analytical_map = {
        (1, 0): psi_10,
        (2, 1): psi_21,
        (2, 0): psi_20,
        (3, 2): psi_32,
        (3, 0): psi_30,
        (3, 1): psi_31,
        (4, 0): psi_40,
        (4, 1): psi_41
    }

    if r.ndim == 1:
        r = r.unsqueeze(-1)
    r_np = r.squeeze().cpu().detach().numpy()

    # Restrict domain from 0 to 35
    r_torch = torch.tensor(r_np).unsqueeze(-1).to(r.device)

    df = pd.read_csv(csv_file)

    num_plots = len(df)
    rows, cols = 2, 4
    fig, axs = plt.subplots(rows, cols, figsize=(16, 6))
    axs = axs.flatten()

    for i, (_, row) in enumerate(df.iterrows()):
        if i >= rows * cols:
            break  # Avoid indexing error

        n, l = int(row['n']), int(row['l'])
        energy = float(row['energy'])
        param_file = row['param_file']
        model_file = os.path.join(model_path, param_file)

        if not os.path.exists(model_file):
            print(f"Model file not found: {model_file}")
            continue

        model = model_class(hidden_layers)
        state_dict = torch.load(model_file, map_location=r.device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(r.device)
        model.eval()

        with torch.no_grad():
            psi_pred = model(r_torch, l).squeeze().cpu().numpy()

        if psi_pred[0] < 0:
            psi_pred = -psi_pred

        ax = axs[i]
        if (n, l) in analytical_map:
            psi_true = analytical_map[(n, l)](r_np)
            ax.plot(r_np, psi_true/r_np, color='orange', label="Analytical",linewidth=0.5)
        ax.plot(r_np, psi_pred/r_np, color='red', linestyle='-.', label="Predicted",linewidth=0.5)
        ax.set_title(fr"$R_{{{n}{l}}}$", fontsize=14)
        ax.tick_params(axis='both', labelsize=8)

    # Hide unused subplot if 7 plots only
    if num_plots < len(axs):
        for j in range(num_plots, len(axs)):
            fig.delaxes(axs[j])

    # Common axis labels
    fig.supxlabel("r", fontsize=16, fontweight="bold")
    fig.supylabel(r"R", fontsize=16, fontweight="bold")

    # Shared legend
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    os.makedirs(plot_path, exist_ok=True)
    plt.savefig(os.path.join(plot_path, "all_wavefunctions.png"), dpi=300)
    plt.close()
