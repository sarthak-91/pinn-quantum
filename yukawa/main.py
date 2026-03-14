import numpy as np 
import torch 
from scripts.network import * 
from scripts.training import * 
from scripts.visuals import * 
from scripts.load_store import * 
from scripts.loss import * 
from scripts.config import device


if __name__ == "__main__":
    hidden_layers = [50]
    wave_form= NN(hidden_layers)
    wave_form = wave_form.to(device)
    n = 1
    l = 0
    N = 2000
    g = 0.025
    r_min = 0.01
    r_max_map = {1:20,2:30,3:45,4:55}
    r = np.linspace(r_min,r_max_map[n],N)
    r_torch = torch.tensor(r, dtype=torch.float32).unsqueeze(1).requires_grad_(True).to(device)
    if n>1:
        excited = True
    else:
        excited=False
    wf_list=[]
    if excited:
        wf_list=load_wavefunctions_for_ortho(n=n,l=l,r=r_torch,hidden_layers=hidden_layers,model_class=NN, print_=True)
        wf_list = [(psi.to(device), E) for psi, E in wf_list]

    energy,lowest_residual,epoch_, time_taken = train_patience(wave_form, r_torch=r_torch, 
                   epochs=600000,loss_fn=loss_fn_rayleigh, 
                   excited_state=excited,lr=1e-3,
                   wf_list=wf_list,n=n, l=l,g=g)
    print("Energy acquired = ",energy.item())
    store(wave_form, energy=energy,lowest_residual = lowest_residual,r=r_torch, n=n, l=l,g=g,epoch_=epoch_,time_ = time_taken)