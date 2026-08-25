# data/

This folder is for **cached/processed** data only (e.g. a cleaned
Parquet file saved after running the loader + cleaning steps, so
future runs don't have to re-read the large Excel file every time).

**The raw dataset itself stays in `course/`** (alongside the course
PowerPoints, where it already was when this project started -
`course/Further Consolidated Data, HnL.xlsx`). This folder does not
duplicate it. `src/data_loader.py`'s `DEFAULT_DATA_PATH` points there.

Nothing is written here yet - this folder exists as a placeholder for
Phase 2 (when the reusable ML framework starts actually processing
data), not because anything currently generates files into it.
