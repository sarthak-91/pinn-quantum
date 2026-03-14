import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
import os
import csv
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams["font.family"] = "Times New Roman"
#plt.rc('text', usetex=True)
#plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

def calculate_potential_term(r):
    V0 = np.sqrt(2)
    g = 0.025
    alpha = g * V0
    potential = -1.0 * V0 * np.exp(-alpha * r) / r
    return sparse.diags(potential)

def calculate_angular_term(r, l):
    angular = l * (l + 1) / (2.0 * r**2)
    return sparse.diags(angular)

def calculate_laplace_three_point(r, N):
    h = r[1] - r[0]
    main_diag = -2.0 / h**2 * np.ones(N)     
    off_diag  =  1.0 / h**2 * np.ones(N - 1)
    return sparse.diags([main_diag, off_diag, off_diag], (0, -1, 1))

def build_hamiltonian(r, l):
    N = len(r)
    laplace_term   = calculate_laplace_three_point(r, N)
    angular_term   = calculate_angular_term(r, l)
    potential_term = calculate_potential_term(r)
    H = -0.5 * laplace_term + angular_term + potential_term
    return H

def normalize_wavefunction(wf, dr):
    """Normalize reduced radial wavefunction u(r)."""
    norm = np.sqrt(np.sum(np.abs(wf)**2) * dr)
    return wf / norm

def load_predicted(r,n,l):
        import torch
        from scripts.network import NN
        hidden_layers = [50]
        r_torch = torch.tensor(r, dtype=torch.float32).unsqueeze(1)
        model_file = f"models/state_n_{n}_l_{l}.pt"
        model = NN(hidden_layers)
        state_dict = torch.load(model_file, map_location=r.device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(r.device)
        model.eval()

        with torch.no_grad():
            psi_pred = model(r_torch, l).squeeze().cpu().numpy()

        if psi_pred[0] < 0:
            psi_pred = -psi_pred
        return psi_pred


def plot_on_ax(ax, r, wf, n, l):
    """
    Plots the wavefunction for a given state (n, l) on a specific subplot axis.
    """
    psi_pred = load_predicted(r,n,l)
    ax.plot(r, wf, label=f'Numerical',color='orange')
    ax.plot(r, psi_pred, label='PINN', linestyle='--',color='red')
    ax.set_title(r'$n = {}, l = {}$'.format(n,l), fontsize=16, fontweight='bold')
    
    x_lim_map = {1:20,2:20,3:40}
    x_ticks_map = {1:10,2:10,3:20}
    ax.set_xlim(0, x_lim_map[n])
    ax.set_xticks(np.arange(0, x_lim_map[n]+1, x_ticks_map[n]))
    y_ticks_start = {(1,0):0, (2,1):0, (3,2):0, (2,0):-0.5, (3,0):-0.25, (3,1):-0.25}

    y_ticks_gap = {(1,0):0.5, (2,1):0.5, (3,2):0.25, (2,0):0.5, (3,0):0.25, (3,1):0.25}
    ax.set_yticks(np.arange(y_ticks_start[(n,l)], np.max(wf)*1.1, y_ticks_gap[(n,l)]))
    ax.tick_params(axis='both', which='major', labelsize=14)




def solve_and_store(r, l,state_ax_map):
    H = build_hamiltonian(r, l)
    number_of_eigenvalues = max(state_mapping[l])  
    eigenvalues, eigenvectors = eigs(H, k=number_of_eigenvalues, which='SR')

    idx = eigenvalues.real.argsort()
    eigenvalues = eigenvalues[idx].real
    eigenvectors = eigenvectors[:, idx]

    for i, n_val in enumerate(state_mapping[l]):
        wf = eigenvectors[:, i].real
        wf = normalize_wavefunction(wf, dr) 
        if wf[0] < 0: 
            wf = -wf
        
        energy = round(eigenvalues[i], 8)
        current_state = (n_val, l)
        if current_state in state_ax_map:
            ax = state_ax_map[current_state]
            plot_on_ax(ax, r, wf, n_val, l)



N = 5000
r_max = 50.0
r = np.linspace(0.01, r_max, N)  
dr = r[1] - r[0]

state_mapping = {
    0: [1, 2, 3],   
    1: [2, 3],      
    2: [3]          
}
target_states = [(1, 0), (2, 0), (3, 0), (2, 1), (3, 1), (3, 2)]
fig, axes = plt.subplots(2, 3, figsize=(18, 6))
state_ax_map = {state: ax for state, ax in zip(target_states, axes.flatten())}
for l in [0, 1, 2]:
    solve_and_store(r,l,state_ax_map)
handles, labels = axes[0, 0].get_legend_handles_labels()
for ax in axes.flatten():
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    ax.legend().remove()

fig.text(0.5, 0.03, 'r', ha='center', fontsize=20, fontweight='bold')
fig.text(0.09, 0.5, r'$rR_{\boldsymbol{nl}}$', va='center', rotation='vertical', fontsize=20)

fig.legend(
    handles,
    labels,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.05),  # push legend higher
    ncol=2,
    frameon=False,
    fontsize=20
)
fig.subplots_adjust(hspace=0.3)
plt.savefig("yukawa_wavefunctions_test.png", dpi=300, bbox_inches="tight")