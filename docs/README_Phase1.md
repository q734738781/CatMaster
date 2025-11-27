# CatMaster Phase 1 - Jobflow-Remote Integration

## Overview

This is the Phase 1 implementation of CatMaster, rebuilt around jobflow-remote v0.1.8 for executing real VASP calculations on CPU clusters and MACE calculations on GPU servers.

## Key Features

### Real Implementations
- **VASP Execution**: Real VASP calculations via jobflow-remote with Slurm support
- **MACE Integration**: GPU-accelerated MACE calculations for structure relaxation and global search
- **Jobflow-Remote**: Full integration with jobflow-remote for job submission and monitoring
- **Authentication**: Support for both SSH key and password authentication
- **Resource Management**: Automatic GPU selection based on available memory

### Removed Mock Implementations
- Replaced all `_simulate_work()` functions with real calculations
- Removed fake DFT energy calculations
- Replaced mock GPU tools with actual MACE implementations

## Demo Scripts

### 1. VASP O2 Molecule Relaxation (CPU)
```bash
python demo_scripts/run_cpu_jobflow_demo.py \
    --structure demo_scripts/assets/POSCAR_O2_BOX \
    --project catmaster-demo \
    --worker slurm-cpu \
    --relax
```

This will:
- Prepare VASP inputs with MPRelaxSet (ISIF=2 for gas molecules)
- Submit to Slurm cluster via jobflow-remote
- Run real VASP relaxation

### 2. MACE O2 Molecule Relaxation (GPU)
```bash
python demo_scripts/run_gpu_jobflow_demo.py \
    --structure demo_scripts/assets/POSCAR_O2_BOX \
    --project catmaster-demo \
    --worker local-gpu \
    --task-type gpu.mace_relax
```

This will:
- Use MACE-MP medium model for relaxation
- Automatically select available GPU
- Run locally on GPU server

## Environment Setup

### PC Control Node
```bash
conda create -n catmaster python=3.12
conda activate catmaster
pip install -r requirements/pc.txt
```

### CPU Cluster
```bash
pip install -r requirements/cpu.txt
export PYTHONPATH=/path/to/catmaster:$PYTHONPATH
export VASP_STD_BIN=/path/to/vasp_std
export VASP_PP_PATH=/path/to/potpaw_PBE.54
```

### GPU Server
```bash
pip install -r requirements/gpu.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
export PYTHONPATH=/path/to/catmaster:$PYTHONPATH
```

## Configuration

### 1. Jobflow-Remote Project
Copy and edit the example configuration:
```bash
cp configs/jobflow_remote/projects/catmaster_demo.yaml ~/.jobflow_remote/projects/
```

Update:
- MongoDB connection strings
- SSH credentials
- Resource allocations
- Python paths

### 2. Secrets Management
```bash
cp configs/.orchestrator/secrets.example.yaml ~/.orchestrator/secrets.yaml
chmod 600 ~/.orchestrator/secrets.yaml
```

Add your:
- CPU cluster password (if using password auth)
- Materials Project API key
- MongoDB credentials

### 3. Start Runners
On CPU cluster:
```bash
jfr runner start slurm-cpu -p catmaster-demo
```

On GPU server:
```bash
jfr runner start local-gpu -p catmaster-demo
```

## Architecture

### CPU (VASP) Flow
1. `prepare_vasp_inputs()` generates INCAR/POSCAR/KPOINTS/POTCAR
2. `build_relax_flow()` creates jobflow Flow with `run_vasp_job()`
3. jobflow-remote submits to Slurm
4. Real VASP execution via subprocess

### GPU (MACE) Flow
1. `build_gpu_flow()` creates Flow with `run_gpu_tool()`
2. Tool registry resolves to `mace_relax()` or `global_search()`
3. MACE calculations using `mace-torch` with GPU acceleration
4. Results saved as VASP format structures

## Updated Components

### Real Implementations
- `/catmaster/gpu/tools.py`: Real MACE relaxation and global search
- `/catmaster/gpu/algorithms/`: MACE algorithms for relaxation and minima hopping
- `/catmaster/pc/tools/cpu_flow/build_flow.py`: Real VASP execution via subprocess

### Configuration
- `/configs/jobflow_remote/projects/catmaster_demo.yaml`: Complete jobflow-remote configuration
- `/requirements/`: Updated dependencies for all components

### Documentation
- `/docs/环境配置指南.md`: Complete deployment guide with Phase 1 features
- This README: Phase 1 implementation summary

## Testing

### Check Installation
```bash
# Verify jobflow-remote
jfr project list

# Test VASP inputs generation
python -c "from catmaster.pc.tools.cpu_flow.prepare_inputs import prepare_vasp_inputs; print('OK')"

# Test GPU tools
python -c "from catmaster.gpu.tools import GPU_TOOL_REGISTRY; print(list(GPU_TOOL_REGISTRY.keys()))"
```

### Monitor Jobs
```bash
# Check job status
jfr job list -p catmaster-demo

# View job details
jfr job info <job_id> -p catmaster-demo
```

## Next Steps

With Phase 1 complete, the system is ready for:
- Production VASP calculations
- MACE-accelerated structure optimization
- Integration with retrieval tools
- LLM-driven task orchestration

All mock implementations have been removed and replaced with real functionality built on jobflow-remote v0.1.8.
