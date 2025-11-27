# CatMaster Demo Scripts

This directory contains demo scripts for testing the CatMaster workflow.

## Understanding Device Types

- **local**: Tools that run locally on your machine
  - `create_molecule`: Creates molecular structures
  - `vasp_prepare`: Prepares VASP input files locally
  - `vasp_summarize`: Analyzes VASP output files locally

- **gpu-worker**: Tools that run on remote GPU nodes (via jobflow-remote)
  - `mace_relax`: Machine learning potential optimization using MACE

- **cpu-worker**: Tools that run on remote CPU/HPC nodes (via jobflow-remote)
  - `vasp_execute`: Runs VASP calculations on HPC clusters

## Demo Scripts

1. **demo_o2_dft_calculation.py**
   - Full workflow for O2 molecule DFT calculation
   - Requires remote execution capabilities (gpu-worker and cpu-worker)
   - Creates structure → ML pre-optimization → VASP optimization → VASP static → Summary

2. **demo_o2_local_only.py**
   - Local-only workflow (no remote execution)
   - Creates O2 structure and prepares VASP inputs
   - Good for testing without remote infrastructure

3. **test_tool_registry.py**
   - Displays all registered tools and their parameters
   - Verifies tool registry configuration

4. **test_tools_manually.py**
   - Tests individual tools without LLM workflow
   - Good for debugging specific tool issues

## Running the Demos

```bash
# Full DFT workflow (requires remote setup)
python demo_o2_dft_calculation.py

# Local-only testing
python demo_o2_local_only.py

# Check tool registry
python test_tool_registry.py
```

## Note on Remote Execution

The "cpu-worker" and "gpu-worker" refer to remote execution via jobflow-remote, not local CPU/GPU execution. These require proper configuration of jobflow-remote with access to HPC resources.
