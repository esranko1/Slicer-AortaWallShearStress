"""
Converts the lab's .mat file into the a/b/c/d .npz format that
5.Point_DeepONet/main.py and 6.PointNet_Velocity/main.py expect to load.

Target shapes:
    a -> (100, 4096, 53)  : xyz (3 cols) + inlet velocity waveform (50 cols),
                            waveform broadcast identically across all 4096 points per patient
    b -> (100, 4096, 1)   : SWSS
    c -> (100,)           : patient identifiers
    d -> (100, 4096, 1)   : TAWSS (time-averaged WSS) — auxiliary multi-task target
"""

import os
import numpy as np
from scipy.io import loadmat


MAT_FILE_PATH = "C:\\Users\\esranko1\\Desktop\\WSS_project\\dataForPy9.mat"

# --- Where the converted file gets saved ------------------------------------
# This matches the existing Data/Sampled folder already in this repo.
OUTPUT_DIR = os.path.join("..", "Data", "Sampled")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "Rpt0_N4096.npz")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Step 1: load the .mat file -----------------------------------------
    mat = loadmat(MAT_FILE_PATH)
    X = mat["X"]                # shape (100, 4096, 3)
    Z = mat["Z"]                 # shape (100, 50)
    SWSS = mat["SWSS"]           # shape (100, 4096)
    TAWSS = mat["TAWSSreal"]     # shape (100, 4096)

    print("Loaded shapes:")
    print("  X:", X.shape)
    print("  Z:", Z.shape)
    print("  SWSS:", SWSS.shape)
    print("  TAWSS:", TAWSS.shape)

    # --- Step 1.5: center each patient's point cloud on its own centroid ----
    # Patients' raw coordinates are not co-registered to a common frame (confirmed
    # empirically — the same grid index landed in wildly different absolute
    # positions across patients, e.g. Z-coordinate split into two clusters ~70-80mm
    # apart). Centering removes that translation inconsistency before the model
    # ever sees it.
    X_centroids_before = X.mean(axis=1)
    print("\nCentroids before centering (first 5 patients):")
    print(X_centroids_before[:5])

    X = X - X.mean(axis=1, keepdims=True)

    X_centroids_after = X.mean(axis=1)
    print("\nCentroids after centering (first 5 patients, should be ~0,0,0):")
    print(X_centroids_after[:5])

    # --- Step 2: broadcast Z across all 4096 points -------------------------
    # Z is (100, 50) — one waveform per patient.
    # You need Z_broadcast with shape (100, 4096, 50), where
    # Z_broadcast[i, j, :] == Z[i, :] for every j (0..4095).
    # Hint: insert a new axis of size 1, then np.repeat along that axis.
    Z_broadcast = np.repeat(Z[:, np.newaxis, :], 4096, axis=1)

    # --- Step 3: concatenate X and Z_broadcast into `a` ---------------------
    # Result should be shape (100, 4096, 53): columns 0-2 = xyz, columns 3-52 = velocity.
    # Hint: np.concatenate(..., axis=-1)
    a = np.concatenate([X, Z_broadcast], axis=-1)

    # --- Step 4: reshape SWSS into `b` --------------------------------------
    # (100, 4096) -> (100, 4096, 1)
    b = SWSS[:, :, np.newaxis]

    # --- Step 5: build patient identifiers `c` ------------------------------
    # Simplest option: np.array([str(i) for i in range(100)])
    # Or reuse your existing patient ID / sno array if you want real IDs.
    c = np.array([str(i) for i in range(100)])

    # --- Step 6: reshape TAWSS into `d` --------------------------------------
    # (100, 4096) -> (100, 4096, 1), auxiliary multi-task target.
    d = TAWSS[:, :, np.newaxis]

    # --- Sanity checks before saving ----------------------------------------
    print("\nFinal shapes (check these before saving):")
    print("  a:", None if a is None else a.shape, "-> expected (100, 4096, 53)")
    print("  b:", None if b is None else b.shape, "-> expected (100, 4096, 1)")
    print("  c:", None if c is None else c.shape, "-> expected (100,)")
    print("  d:", None if d is None else d.shape, "-> expected (100, 4096, 1)")

    # --- Step 7: save --------------------------------------------------------
    np.savez_compressed(OUTPUT_PATH, a=a, b=b, c=c, d=d)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
