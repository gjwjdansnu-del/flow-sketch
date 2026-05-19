# macOS Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate flow_sketch
```

Verify SU2:

```bash
SU2_CFD --help
```

Verify gmsh:

```bash
gmsh --version
```

Optional project tool check:

```bash
python cfd_pipeline/scripts/check_tools.py
```
