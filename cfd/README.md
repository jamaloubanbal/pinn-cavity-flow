# CFD Cases Layout

This directory contains CFD references used by `src/compare.py`.

## Tracked in git

- `openfoam_cavity/system/`, `openfoam_cavity/constant/`, `openfoam_cavity/0/`
- `cs_cavity/cavity/DATA/`, `cs_cavity/cavity/SRC/`
- helper scripts (for example `cs_cavity/generate_mesh.py`)

## Not tracked in git

Generated solver outputs are ignored by `.gitignore`:

- OpenFOAM runtime folders (`openfoam_cavity/15`, `20`, `25`, ...)
- OpenFOAM `postProcessing/`, `constant/polyMesh/`, and solver logs
- Code_Saturne `cavity/RESU/` and generated `cavity.msh`

Regenerate these locally before running `python3 src/compare.py`.
