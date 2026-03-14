import pandas as pd

def average_results(meta_csv, out_csv=None):
    """
    Reads metadata CSV, computes averages grouped by (n,l,g),
    keeps one entry per state, and saves if requested.
    """
    # load CSV safely
    df = pd.read_csv(meta_csv, index_col=False)

    # compute averages (one row per state)
    df_avg = df.groupby(["n", "l", "g"], as_index=False)[
        ["energy", "pde_loss", "epochs", "time"]
    ].mean()

    # if you also want to keep param_file and wf_file, pick the *first* one per state
    df_files = df.groupby(["n", "l", "g"], as_index=False)[
        ["param_file", "wf_file"]
    ].first()

    # merge averages with file references (now one row per state)
    df_out = pd.merge(df_avg, df_files, on=["n", "l", "g"], how="left")

    # optionally save
    if out_csv:
        df_out.to_csv(out_csv, index=False)

    return df_out




# ---- Example usage ----
if __name__ == "__main__":
    averaged = average_results("hbc2.csv", "avg_registry.csv")
    print(averaged.head())
