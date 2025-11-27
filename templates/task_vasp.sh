#!/bin/bash

#SBATCH -J Manual_Relexation
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --export=ALL


### Set intel environment###
source /public/home/chenhh/intel/oneapi/setvars.sh --config="$HOME/oneapi-nopy.cfg"
cd $SLURM_SUBMIT_DIR
/public/home/chenhh/dos2unix INCAR
mpirun -n $SLURM_NPROCS /public/home/chenhh/vasp_program/vasp6.4.3-vtst/bin/vasp_std