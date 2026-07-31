"""
Preprocess geometric features for all patients and save as .npy file.

This script:
1. Loads the MATLAB data
2. Computes normals and distances using geometric_features.py
3. Saves the enriched features as X_with_features.npy

Usage:
    python preprocess_geometric_features.py --input dataForPy9.mat --output X_with_features.npy
"""

import argparse
import numpy as np
import scipy.io
from geometric_features import process_point_cloud
import os


def main():
    parser = argparse.ArgumentParser(description='Preprocess geometric features for all patients.')
    parser.add_argument('--input', type=str, default='dataForPy9.mat', help='Path to input .mat file')
    parser.add_argument('--output', type=str, default='X_with_features.npy', help='Path to output .npy file')
    parser.add_argument('--inlet_idx', type=int, default=587, help='Inlet vertex index')
    parser.add_argument('--k_neighbors', type=int, default=10, help='Number of neighbors for normal estimation')
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.input}...")
    mat_data = scipy.io.loadmat(args.input)
    X = mat_data['X']  # shape: (100, 4096, 3)
    print(f"  Data shape: {X.shape}")

    # Preprocess all patients
    print(f"\nPreprocessing {X.shape[0]} patients...")
    X_features = []

    for i in range(X.shape[0]):
        if (i + 1) % 10 == 0:
            print(f"  Patient {i + 1}/{X.shape[0]}...")

        point_cloud = X[i]  # shape: (4096, 3)
        features = process_point_cloud(
            point_cloud,
            inlet_vertex_idx=args.inlet_idx,
            k_neighbors=args.k_neighbors
        )
        X_features.append(features)

    X_features = np.array(X_features)  # shape: (100, 4096, 7)
    print(f"\nFeatures shape: {X_features.shape}")

    # Save to disk
    print(f"Saving to {args.output}...")
    np.save(args.output, X_features)
    print(f"✓ Done!")
    print(f"\nTo use these features in training:")
    print(f"  python 3.PointNet/main.py --use_geometric_features --geometric_features_path {args.output} -ic xyznd ...")
    print(f"\nNote: -ic xyznd means use coordinates (xyz), normals (n), and distance (d)")


if __name__ == "__main__":
    main()
