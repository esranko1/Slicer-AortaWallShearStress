"""
Pulls the per-patient pulse wave velocity out of dataForPy9.mat for main.py.

Uses PWVreal (the "real" variants are the corrected ones, as for Xreal / Zreal /
SWSSreal in 1.Data_preprocessing/1.3.convert_mat_to_npz.py). Rows are in the
same patient order as Rpt0_N4096.npz, whose identifiers are "0".."99"; the
same identifiers are stored here and main.py checks they match.

Usage (from this folder):
    python extract_pwv.py
    python extract_pwv.py --mat "C:\\path\\to\\dataForPy9.mat" --variable PWV
Writes ../Data/Sampled/pwv_N100.npz with pwv (100,) in m/s and c (ids).
"""

import argparse

import numpy as np
from scipy.io import loadmat

DEFAULT_MAT = "C:\\Users\\esranko1\\Desktop\\WSS_project\\dataForPy9.mat"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mat", default=DEFAULT_MAT)
    ap.add_argument("--variable", default="PWVreal")
    ap.add_argument("--output", default="../Data/Sampled/pwv_N100.npz")
    args = ap.parse_args()

    mat = loadmat(args.mat)
    if args.variable not in mat:
        names = [k for k in mat if not k.startswith("__")]
        raise SystemExit(f"{args.variable} not in {args.mat}; variables: {names}")
    pwv = np.asarray(mat[args.variable], dtype=np.float64).ravel()
    print(f"{args.variable}: {pwv.shape[0]} values, min {np.nanmin(pwv):.2f}, median {np.nanmedian(pwv):.2f}, "
          f"max {np.nanmax(pwv):.2f}, missing {np.isnan(pwv).sum()}")
    if np.isnan(pwv).any():
        raise SystemExit("PWV has missing values; decide how to handle them before training.")

    ids = np.array([str(i) for i in range(pwv.shape[0])])  # same as 1.3.convert_mat_to_npz.py's c
    np.savez_compressed(args.output, pwv=pwv.astype(np.float32), c=ids, source=f"{args.mat}:{args.variable}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
