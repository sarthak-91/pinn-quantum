import torch 
import os 
import numpy as np 
import pandas as pd
from scripts.visuals import *
from scripts.load_store import log_errors
import time 


def update_lr(optimizer, pde_loss, base_lr=1e-3, min_lr=1e-7, max_loss=5e-5, min_loss=5e-8):
    clamped_loss = max(min(pde_loss.item(), max_loss), min_loss)
    alpha = (clamped_loss - min_loss) / (max_loss - min_loss)
    new_lr = min_lr + alpha * (base_lr - min_lr)
    for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    
    return new_lr



def train(model, r_torch, loss_fn, epochs=300000, excited_state=False, wf_list=[], 
          n=None, l=0, g=0.002, lr=1e-3, 
          window_size=100, delta=1e-8, patience=3):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    energies = []
    pde_errors = []
    ortho_errors = []
    norm_errors = []



    start_time = time.time()
    for epoch in range(epochs):
        total_loss = torch.zeros(1, device=r_torch.device)
        optimizer.zero_grad()

        energy, pde_loss, norm_loss, ortho_loss, pos_loss = loss_fn(
            model, r_torch, excited_state, wf_list=wf_list, n=n, l=l,
        )


        # Weighted total loss
        total_loss += 1.0 * energy
        total_loss += 10.0 * pde_loss
        total_loss += 1.0 * norm_loss
        total_loss += 10.0 * ortho_loss
        total_loss += 10.0 * pos_loss

        """weights for n == 4: 0.1,1,10,10,10
            weights for everything else: 1,10,1,1,10"""

        total_loss.backward()
        optimizer.step()

        losses.append(total_loss.item())
        energies.append(energy.item())
        pde_errors.append(pde_loss.item())
        norm_errors.append(norm_loss.item())
        ortho_errors.append(ortho_loss.item())

        # Learning rate adjustment 
        current_lr =  lr if n == 1 else update_lr(optimizer, pde_loss)

        # Energy-based convergence check
        if pde_loss.item() < 5e-8 and norm_loss.item()<1e-8 and ortho_loss.item()<1e-8:
                print(f"\nConverged at epoch {epoch}")
                print(f"Final energy: {energy.item():.8f} | PDE: {pde_loss.item():.8e} | Norm: {norm_loss.item():.8e} | Ortho: {ortho_loss.item():.8e}")
                break



        # Logging
        if epoch % 5000 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:5d} | Energy: {energy.item():.8f} |", end=' ')
            print(f"PDE: {pde_loss.item():.8f} | Norm: {norm_loss.item():.8f} |", end=' ')
            print(f"Boundary: {pos_loss.item():.8f}",end=' ')
            if excited_state:
                print(f"Ortho: {ortho_loss.item():.8f} |", end=' ')
            print(f"LR: {current_lr:.2e}")

    elapsed_time = time.time() - start_time



    # Logging and Plotting
    log_errors(n=n, l=l, energy_list=energies, 
               norm_list=norm_errors, ortho_list=ortho_errors, pde_list=pde_errors)
    plot_loss_curve(losses, save_path="training.png")
    plot_energy_curve(energies, n=n, l=l, path="convergence")

    return energy, pde_loss.item(), epoch, elapsed_time



