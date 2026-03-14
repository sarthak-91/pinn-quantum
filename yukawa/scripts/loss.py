import os 
import pandas as pd
import torch 
from scripts.diff import *
from scripts.load_store import load_wavefunctions_for_ortho
from scripts.config import device

def ortho_loss_fn_wave(psi, wf_list, r_tensor):
    dr = r_tensor[1] - r_tensor[0]
    ortho_loss = torch.zeros(1, device=r_tensor.device)
    for psi_ in wf_list:
        dot = torch.sum(psi.squeeze() * psi_.squeeze() * dr) ** 2
        ortho_loss += dot
    return ortho_loss

def norm_loss_fn(psi,r_tensor):
    dr = r_tensor[1] - r_tensor[0]
    norm = torch.sum(psi ** 2 * dr)
    norm_loss = (1 - norm)**2
    return norm_loss


def loss_fn_rayleigh(model, r_tensor, excited_state=False,wf_list=[], n=None, l=0,g=0.002):
    """
    Loss function with Rayleigh quotient 
    E = <psi | H | psi> / <psi | psi>
    """
    psi = model(r_tensor, l)
    psi_dd = gradient(psi, r_tensor, 2)
    dr = r_tensor[1] - r_tensor[0]
    ortho_loss = torch.zeros(1, device=r_tensor.device)
    ortho_operator = torch.zeros_like(psi, device=r_tensor.device)

    if excited_state and n is not None and wf_list != []:
        ortho_loss = ortho_loss_fn_wave(psi, [wf for wf, _ in wf_list], r_tensor)

    V = torch.sqrt(torch.tensor(2.0))
    alpha = g*V
    yukawa_potential = V* torch.exp(-alpha*r_tensor)
    H_psi = -0.5 * psi_dd - psi *yukawa_potential/ r_tensor + l * (l + 1) * psi / (2 * r_tensor ** 2)
    #H_psi -= ortho_operator

    norm = torch.sum(psi ** 2 * dr)
    psi_H_psi = psi * H_psi
    energy = torch.sum(psi_H_psi * dr)/norm

    norm_loss = (1 - norm)**2
    positive_loss = (psi[0][0] - 1e-3)** 2 + psi[-1][0] ** 2 
    residual = H_psi - energy * psi
    pde_loss = torch.mean(residual**2)
    
    return energy, pde_loss,norm_loss, ortho_loss,positive_loss

