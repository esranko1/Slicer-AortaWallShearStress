# Results Log

Tracking `main.py` runs across the `SWSS`/`Xreal`/`SWSSreal` data-source fix
([1.3.convert_mat_to_npz.py](../1.Data_preprocessing/1.3.convert_mat_to_npz.py)).
Velocity encoder (dense-layer fusion) is present in all runs below.

## Current (correct data: Xreal / Zreal / SWSSreal)

| Region | Patient Pearson | Region Spearman | Whole-vessel Pearson | Whole-vessel Spearman | CCC | Notes |
|---|---|---|---|---|---|---|
| Proximal Ascending (pasc) | 0.901 | 0.752 | 0.671 | 0.644 | 0.600 | current project focus |
| Thoracic Arch (arch) | 0.752 | 0.714 | 0.640 | 0.652 | 0.611 | current project focus for this run |
| Descending Aorta (desc) | 0.721 | 0.658 | 0.724 | 0.671 | 0.681 | current project focus for this run |
| Abdominal Aorta (abda) | — | — | — | — | — | pending |
| Composite (stitched) | — | — | — | — | — | pending — run `stitch_specialists.py` once all 4 done |

## Superseded (incorrect data: plain SWSS/X/Z — do not use for reporting)

| Region | Patient Pearson | Region Spearman | Notes |
|---|---|---|---|
| Proximal Ascending (pasc) | 0.897 | 0.751 | wrong data |
| Thoracic Arch (arch) | 0.770 | 0.719 | wrong data |
| Descending Aorta (desc) | — | — | never run standalone before the fix |
| Abdominal Aorta (abda) | — | — | never run standalone before the fix |
| Composite (stitched) | 0.816 (whole-vessel Pearson) | 0.777 | wrong data |
