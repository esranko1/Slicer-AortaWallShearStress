"""
Builds the train/valid split file that main.py loads separately from the
main data file. Must contain patient identifiers matching whatever you put
in `c` in 1.3.convert_mat_to_npz.py.

main.py looks for this at (with --split_method random, --N_samples 100):
    ../data/npy/combined_100_split_random_train_valid.npz
Since this repo doesn't have that data/npy folder convention, save it
somewhere sensible under Data/ and point --dir_base_load_data / the split
path accordingly when you run main.py.
"""

import os
import numpy as np

OUTPUT_DIR = os.path.join("..", "Data")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "combined_100_split_random_train_valid.npz")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- All patient identifiers ---------------------------------------------
    # Must exactly match the values you used for `c` in 1.3.convert_mat_to_npz.py.
    all_ids = None  # TODO: e.g. np.array([str(i) for i in range(100)])

    # --- Decide which patients go to train vs. valid -------------------------
    # Reuse your existing patient-grouping logic here (the one that avoids
    # putting the same patient in both train and validation) rather than a
    # plain random split.
    train_ids = None  # TODO
    valid_ids = None  # TODO

    print("Split sizes (check before saving):")
    print("  train:", None if train_ids is None else len(train_ids))
    print("  valid:", None if valid_ids is None else len(valid_ids))

    # --- Save ------------------------------------------------------------------
    # Uncomment once train_ids/valid_ids are filled in.
    # np.savez_compressed(OUTPUT_PATH, train=train_ids, valid=valid_ids)
    # print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
