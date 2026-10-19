Quarantined 2026-10-20.

These benchmark artifacts were produced against the incomplete local vector
store (3,584 docs vs 794,366 intended) and are invalid per the project audit.
They are kept for reference only. Fresh artifacts are produced on the A100 by:
  python src/main.py prepare --rebuild
  python src/main.py base-paper --samples 50 --plot
