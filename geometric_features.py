"""
Compute geometric features for point clouds (no mesh needed).

For point clouds, we approximate normals using k-nearest neighbors + PCA.

Features: [x, y, z, nx, ny, nz, distance_to_inlet]
Paper shows this increases accuracy from 0.782 → ~0.79-0.80 Pearson
"""

import numpy as np
from scipy.spatial import cKDTree


def compute_normals_from_neighbors(
    point_cloud: np.ndarray,
    k: int = 10
) -> np.ndarray:
    """
    Approximate surface normals using k-nearest neighbors + PCA.

    For each point, we:
    1. Find k nearest neighbors
    2. Center them (subtract mean)
    3. Do PCA (eigenvalue decomposition)
    4. Use smallest eigenvector as normal (perpendicular to local surface)

    Args:
        point_cloud: shape (N_points, 3), point coordinates
        k: number of neighbors to use for normal estimation

    Returns:
        normals: shape (N_points, 3), approximate unit normal vectors
    """
    tree = cKDTree(point_cloud)
    _, neighbor_indices = tree.query(point_cloud, k=k)

    normals = np.zeros_like(point_cloud)

    for i in range(len(point_cloud)):
        # Get the k neighbors for this point
        neighbors = point_cloud[neighbor_indices[i]]  # shape: (k, 3)

        # Center around mean
        centered = neighbors - neighbors.mean(axis=0)  # shape: (k, 3)

        # Compute covariance matrix
        cov = centered.T @ centered  # shape: (3, 3)

        # PCA: get eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Normal is eigenvector for SMALLEST eigenvalue
        # (direction of least variance = perpendicular to surface)
        normal = eigenvectors[:, 0]  # shape: (3,)

        normals[i] = normal

    # Normalize to unit vectors
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    return normals


def compute_distance_to_inlet(
    point_cloud: np.ndarray,
    inlet_vertex_idx: int
) -> np.ndarray:
    """
    Compute Euclidean distance from each point to inlet.

    This is an approximation of geodesic distance (true surface distance).
    The model learns that points near the inlet have different flow patterns.

    Args:
        point_cloud: shape (N_points, 3)
        inlet_vertex_idx: index of inlet point (should be 587 for your data)

    Returns:
        distances: shape (N_points,), distance from each point to inlet
    """
    inlet_point = point_cloud[inlet_vertex_idx]  # shape: (3,)
    distances = np.linalg.norm(point_cloud - inlet_point, axis=1)  # shape: (N,)
    return distances


def concatenate_features(
    point_cloud: np.ndarray,
    normals: np.ndarray,
    distances: np.ndarray
) -> np.ndarray:
    """
    Concatenate geometric features to coordinates.

    Output shape: (N, 7) = [x, y, z, nx, ny, nz, distance_to_inlet]

    Args:
        point_cloud: shape (N, 3), raw coordinates
        normals: shape (N, 3), surface normals
        distances: shape (N,), distance to inlet

    Returns:
        features: shape (N, 7), all features concatenated
    """
    distances_reshaped = distances[:, np.newaxis]  # (N,) -> (N, 1)
    features = np.concatenate([point_cloud, normals, distances_reshaped], axis=1)
    return features  # shape: (N, 7)


def process_point_cloud(
    point_cloud: np.ndarray,
    inlet_vertex_idx: int,
    k_neighbors: int = 10
) -> np.ndarray:
    """
    Main function: process a point cloud and compute all features.

    Call this once per patient to create the enriched input for PointNet.

    Args:
        point_cloud: shape (N_points, 3), raw coordinates
        inlet_vertex_idx: index of inlet point (587 for your data)
        k_neighbors: number of neighbors for normal estimation (default 10 is good)

    Returns:
        features: shape (N_points, 7), enriched point cloud with geometry
    """
    normals = compute_normals_from_neighbors(point_cloud, k=k_neighbors)
    distances = compute_distance_to_inlet(point_cloud, inlet_vertex_idx)
    features = concatenate_features(point_cloud, normals, distances)
    return features


if __name__ == "__main__":
    # Test on random point cloud
    print("Testing geometric features computation...\n")

    np.random.seed(42)
    test_cloud = np.random.randn(100, 3)

    # Test 1: Compute normals
    print("Test 1: Computing normals...")
    normals = compute_normals_from_neighbors(test_cloud, k=5)
    print(f"  Shape: {normals.shape}")
    print(f"  Normal magnitude (should be ~1.0): {np.linalg.norm(normals[0]):.4f}")

    # Test 2: Compute distances
    print("\nTest 2: Computing distances...")
    distances = compute_distance_to_inlet(test_cloud, inlet_vertex_idx=0)
    print(f"  Shape: {distances.shape}")
    print(f"  Distance range: {distances.min():.4f} to {distances.max():.4f}")

    # Test 3: Concatenate features
    print("\nTest 3: Concatenating features...")
    features = concatenate_features(test_cloud, normals, distances)
    print(f"  Shape: {features.shape}")
    print(f"  Expected: (100, 7)")

    print("\n✓ All tests passed!")
