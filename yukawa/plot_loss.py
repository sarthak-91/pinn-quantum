import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import numpy as np # Import numpy for infinity
plt.rcParams["font.family"] = "Times New Roman"

def plot_losses(base_dir="logs"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 4))

    # --- NEW: Variables to track the overall data range ---
    min_pde_loss = np.inf
    max_pde_loss = -np.inf
    min_ortho_loss = np.inf
    max_ortho_loss = -np.inf

    # Iterate through n folders
    for n_folder in sorted(os.listdir(base_dir)):
        n_path = os.path.join(base_dir, n_folder)
        if not os.path.isdir(n_path):
            continue

        # Iterate through log files inside each n folder
        for log_file in sorted(os.listdir(n_path)):
            if not log_file.endswith(".csv"):
                continue

            filepath = os.path.join(n_path, log_file)
            df = pd.read_csv(filepath)
            n_val = int(df["n"].iloc[0])
            l_val = int(df["l"].iloc[0])
            label = f"n={n_val}, l={l_val}"
            threshold = 1e-7
            ortho = df["ortho_loss"].values
            below = np.where(ortho < threshold)[0]

            if len(below) > 0:
                stays_below_from = None
                for idx in below:
                    if np.all(ortho[idx:] < threshold):
                        stays_below_from = idx
                        break
                if stays_below_from is not None:
                    print(f"Ortho loss for {label} stays below {threshold} from step {stays_below_from+1}")
                else:
                    first_below = below[0]
                    print(f"Ortho loss for {label} goes below {threshold} at step {first_below+1} but later rises again")
            else:
                print(f"Ortho loss for {label} never goes below {threshold}")

            # --- NEW: Update the min and max loss values ---
            min_pde_loss = min(min_pde_loss, df["pde_loss"].min())
            max_pde_loss = max(max_pde_loss, df["pde_loss"].max())
            if "ortho_loss" in df.columns:
                min_ortho_loss = min(min_ortho_loss, df["ortho_loss"].min())
                max_ortho_loss = max(max_ortho_loss, df["ortho_loss"].max())

            # Extract state info

            steps = range(1, len(df) + 1)
            ax1.plot(steps, df["pde_loss"], label=label)
            if "ortho_loss" in df.columns:
                ax2.plot(
                    steps[::500],
                    df["ortho_loss"].iloc[::500],
                    label=label
                )

    # Top subplot formatting
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.yaxis.set_major_locator(LogLocator(base=10.0, subs=None))  
    ax1.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(), numticks=10))
    #ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.set_ylabel(r"$\mathcal{L}_D$")
    #ax1.set_xlim(1, 1e6)
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    # --- NEW: Set Y-axis limits to fit all data ---


    # Bottom subplot formatting
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.yaxis.set_major_locator(LogLocator(base=10.0, subs=None))  
    ax2.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(), numticks=10))
    #ax2.grid(True, which="both", ls="--", alpha=0.5)
    #ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.set_ylabel(r"$\mathcal{L}_O$")
    ax2.set_xlim(1, 1e6)
    ax2.set_xlabel("Epochs")



    # Consolidate legends from both axes to remove duplicates
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc='upper center',
        bbox_to_anchor=(0.5, 1),
        ncol=4,
        frameon=False
    )

    plt.subplots_adjust(top=0.88, hspace=0.4)
    plt.savefig("yukawa_loss.png", dpi=500)

if __name__ == "__main__":
    plot_losses()