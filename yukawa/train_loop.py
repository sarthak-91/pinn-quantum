def train(model, r_torch, loss_fn, epochs=300000, excited_state=False, wf_list=[], 
          n=None, l=0, g=0.002, lr=1e-3, 
          window_size=1000, delta=1e-8, patience=3):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    energies = []
    pde_errors = []
    ortho_errors = []
    norm_errors = []

    # Best model tracking
    best_model_state = None
    lowest_residual = float("inf")

    # Convergence logic variables
    count = 0
    prev_avg = float('inf')
    rest = False
    transition_epoch = 0

    start_time = time.time()
    for epoch in range(epochs):
        total_loss = torch.zeros(1, device=r_torch.device)
        optimizer.zero_grad()

        energy, pde_loss, norm_loss, ortho_loss, pos_loss = loss_fn(
            model, r_torch, excited_state, wf_list=wf_list, n=n, l=l, g=g
        )

        # Save best model
        if pde_loss.item() < lowest_residual:
            lowest_residual = pde_loss.item()
            best_model_state = deepcopy(model.state_dict())

        # Weighted total loss
        total_loss += 0.1 * energy
        total_loss += 1.0 * pde_loss
        total_loss += 1.0 * norm_loss
        total_loss += 1.0 * ortho_loss
        total_loss += 10.0 * pos_loss

        total_loss.backward()
        optimizer.step()

        losses.append(total_loss.item())
        energies.append(energy.item())
        pde_errors.append(pde_loss.item())
        norm_errors.append(norm_loss.item())
        ortho_errors.append(ortho_loss.item())

        # Learning rate adjustment (if needed)
        current_lr = update_lr(optimizer, pde_loss)

        # Energy-based convergence check
        if epoch % window_size == 0 and epoch > 0:
            current_avg = np.mean(energies[-window_size:])
            diff = abs(prev_avg - current_avg)
            prev_avg = current_avg

            if not rest and diff <= delta:
                count += 1
            else:
                count = 0

            if count >= patience:
                print(f"\nConverged at epoch {epoch}")
                print(f"Final energy: {energy.item():.8f} | PDE: {pde_loss.item():.8e} | Norm: {norm_loss.item():.8e} | Ortho: {ortho_loss.item():.8e}")
                print("Found lowest residual, stopping training.",lowest_residual)
                #break
        if pde_loss.item() < 1e-7 and norm_loss.item()<1e-8 and ortho_loss<1e-8:
                print(f"\nConverged at epoch {epoch}")
                print(f"Final energy: {energy.item():.8f} | PDE: {pde_loss.item():.8e} | Norm: {norm_loss.item():.8e} | Ortho: {ortho_loss.item():.8e}")
                break


        # Logging
        if epoch % 5000 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:5d} | Energy: {energy.item():.8f} |", end=' ')
            print(f"PDE: {pde_loss.item():.8f} | Norm: {norm_loss.item():.8f} |", end=' ')
            if excited_state:
                print(f"Ortho: {ortho_loss.item():.8f} |", end=' ')
            print(f"LR: {current_lr:.2e}")

    elapsed_time = time.time() - start_time

    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Log & Plot
    log_errors(n=n, l=l, energy_list=energies, 
               norm_list=norm_errors, ortho_list=ortho_errors, pde_list=pde_errors)
    plot_loss_curve(losses, save_path="training.png")
    plot_energy_curve(energies, n=n, l=l, path="convergence")

    return energy, lowest_residual, epoch, elapsed_time
