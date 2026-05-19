#!/bin/bash

#SBATCH -J Manual_Vasp
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=104
#SBATCH --export=ALL


### Set intel environment###
module load  vasp/6.4.3-vtst-sol-oneapi2025.2
cd $SLURM_SUBMIT_DIR
mpirun -n $SLURM_NPROCS vasp_gam