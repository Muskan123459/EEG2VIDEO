# Legacy single-file implementations

These are the original monolithic scripts that the `eeg2video/` package was
refactored from. They are kept here verbatim for reference and reproducibility.

- `dl_new (1).py` — the newer, more complete version (checkpoint resume,
  cosine-annealing LR schedule with warmup, CSV logging, periodic checkpoints,
  and a full metric-evaluation entry point). **This is the version the package
  was refactored from.**
- `dl_new.py` — an earlier version (100-epoch config, `ReduceLROnPlateau`
  scheduling, no resume/logging).

The refactored package reproduces the code in `dl_new (1).py` exactly; only the
file organisation changed. Prefer the package (`eeg2video/`) and the entry-point
scripts (`scripts/`) for all new work.
