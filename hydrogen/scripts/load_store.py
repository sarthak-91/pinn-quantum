import os
import pandas as pd
import torch
from scripts.config import *
import numpy as np 
import csv 
import numpy as np



def find_nearest_state(n, model_path=MODEL_PATH,csv_path=CSV_FILE):
    df = pd.read_csv(csv_path)


    df_filtered = df[df['n'] < n]
    if not df_filtered.empty:
        best_row = df_filtered.sort_values('n', ascending=False).iloc[0]
        return os.path.join(model_path, best_row['param_file'])

    df_higher = df[df['n'] > n]
    if not df_higher.empty:
        best_row = df_higher.sort_values('n').iloc[0]
        return os.path.join(model_path, best_row['param_file'])

    return None

def store(model, r,energy, lowest_residual, n, l, g,epoch_,time_,wave_path=WF_PATH,model_path=MODEL_PATH, csv_file= CSV_FILE):
    os.makedirs(wave_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    param_filename = f"state_n_{n}_l_{l}.pt"
    wf_filename = f"wf_n_{n}_l_{l}.npy"

    torch.save(model.state_dict(), os.path.join(model_path, param_filename))

    r_tensor = r.clone().detach().unsqueeze(1).to(device)
    with torch.no_grad():
        psi = model(r_tensor,l).squeeze().cpu().numpy()
    np.save(os.path.join(wave_path, wf_filename), psi)


    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["n", "l", "g","energy","pde_loss","epochs","time", "param_file", "wf_file"])
        writer.writerow([n, l, g,energy.item(),lowest_residual,epoch_,time_, param_filename, wf_filename])





def log_errors(n, l, energy_list, pde_list, norm_list, ortho_list, logging_path=LOGGING_PATH):
    subdir = os.path.join(logging_path, f"n{n}")
    os.makedirs(subdir, exist_ok=True)

    log_file = os.path.join(subdir, f"log_n{n}_l{l}.csv")
    file_exists = os.path.isfile(log_file)

    with open(log_file, mode='w', newline='') as csvfile:
        fieldnames = ["n", "l", "energy", "pde_loss", "norm_loss", "ortho_loss"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for e, pde, norm, ortho in zip(energy_list, pde_list, norm_list, ortho_list):
            writer.writerow({
                "n": n,
                "l": l,
                "energy": f"{e:.9f}",
                "pde_loss": f"{pde:.9f}",
                "norm_loss": f"{norm:.9f}",
                "ortho_loss": f"{ortho:.9f}"
            })






def load_wavefunctions_for_ortho(n, l, r, wave_path=WF_PATH, csv_path=CSV_FILE, model_path=MODEL_PATH,print_=False, model_class=None,hidden_layers=[10,10]):
    assert model_class is not None, "You must pass `model_class` to instantiate models."

    df = pd.read_csv(csv_path)

    states_to_load_map = {
        (4, 1): [(3, 1), (2, 1)],
        (4, 0): [(3, 0), (2, 0), (1, 0)],
        (3, 2): [],
        (3, 1): [(2, 1)],
        (3, 0): [(2, 0), (1, 0)],
        (2, 1): [],
        (2, 0): [(1, 0)],
        (1, 0): []
    }

    target_states = states_to_load_map.get((n, l), [])

    if not target_states:
        if print_:
            print(f"No orthogonalization states defined for (n={n}, l={l})")
        return []

    wavefunctions = []

    for n_prev, l_prev in target_states:
        row = df[(df['n'] == n_prev) & (df['l'] == l_prev)]
        if row.empty:
            if print_:
                print(f"⚠️  No registry entry found for (n={n_prev}, l={l_prev})")
            continue

        row = row.iloc[0]
        model_file = os.path.join(model_path, row['param_file'])

        if not os.path.exists(model_file):
            if print_:
                print(f"🚫 Model file not found: {model_file}")
            continue

        try:
            model = model_class(hidden_layers) 
            state_dict = torch.load(model_file, map_location=r.device, weights_only=True)
            model.load_state_dict(state_dict)
            model.to(r.device)  
            model.eval()

            with torch.no_grad():
                psi_r = model(r,l_prev).squeeze()

            wavefunctions.append((psi_r, row['energy']))

            if print_:
                print(f"✅ Evaluated: n={n_prev}, l={l_prev}, E={row['energy']:.6f}")

        except Exception as e:
            if print_:
                print(f"❌ Error loading/evaluating model for (n={n_prev}, l={l_prev}): {e}")

    return wavefunctions


def find_nearest_state(n, model_path=MODEL_PATH,csv_path=CSV_FILE):
    df = pd.read_csv(csv_path)


    df_filtered = df[df['n'] < n]
    if not df_filtered.empty:
        best_row = df_filtered.sort_values('n', ascending=False).iloc[0]
        return os.path.join(model_path, best_row['param_file'])

    df_higher = df[df['n'] > n]
    if not df_higher.empty:
        best_row = df_higher.sort_values('n').iloc[0]
        return os.path.join(model_path, best_row['param_file'])

    return None

def fidelity(n, l, psi_nn, r):
    """
    Compute the fidelity (overlap) between psi (NN-predicted) and analytical psi_nl.
    
    Parameters:
        n (int): Principal quantum number
        l (int): Angular momentum quantum number
        psi (callable): Neural network wavefunction, psi(r)
        r (np.ndarray): Radial grid points

    Returns:
        float: Fidelity F = |<psi_NN | psi_analytical>|
    """
    # Analytical radial function from the map
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
    analytical_fn = analytical_map.get((n, l))
    if analytical_fn is None:
        raise ValueError(f"No analytical wavefunction available for (n={n}, l={l})")
    

    psi_analytic = analytical_fn(r)
    

    integrand = psi_nn * psi_analytic
    dr = r[1] - r[0]
    overlap = np.sum(integrand * dr)
    
    return np.abs(overlap.item())

def store(model, r,energy, lowest_residual, n, l,epoch_,time_,wave_path=WF_PATH,model_path=MODEL_PATH, csv_file= CSV_FILE):
    os.makedirs(wave_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    param_filename = f"state_n_{n}_l_{l}.pt"
    wf_filename = f"wf_n_{n}_l_{l}.npy"

    torch.save(model.state_dict(), os.path.join(model_path, param_filename))
    r_tensor = r.clone().detach().unsqueeze(1).to(device)
    with torch.no_grad():
        psi = model(r_tensor,l).squeeze().cpu().numpy()
    F = fidelity(n, l, psi, r_tensor.squeeze().cpu().numpy())
    np.save(os.path.join(wave_path, wf_filename), psi)


    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["n", "l","energy","fidelity","pde_loss","epochs","time", "param_file", "wf_file"])
        writer.writerow([n, l, energy.item(),F,lowest_residual,epoch_,time_, param_filename, wf_filename])





def log_errors(n, l, energy_list, pde_list, norm_list, ortho_list, logging_path=LOGGING_PATH):
    subdir = os.path.join(logging_path, f"n{n}")
    os.makedirs(subdir, exist_ok=True)

    log_file = os.path.join(subdir, f"log_n{n}_l{l}.csv")
    file_exists = os.path.isfile(log_file)

    with open(log_file, mode='w', newline='') as csvfile:
        fieldnames = ["n", "l", "energy", "pde_loss", "norm_loss", "ortho_loss"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for e, pde, norm, ortho in zip(energy_list, pde_list, norm_list, ortho_list):
            writer.writerow({
                "n": n,
                "l": l,
                "energy": f"{e:.9f}",
                "pde_loss": f"{pde:.9f}",
                "norm_loss": f"{norm:.9f}",
                "ortho_loss": f"{ortho:.9f}"
            })

