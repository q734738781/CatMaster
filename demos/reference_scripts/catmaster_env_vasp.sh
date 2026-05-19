export PATH=/public/home/chenhh/anaconda3/condabin:/public/software/vasp.6.4.1-vtst-sol/bin:$PATH
export PYTHONPATH=/public/home/chenhh/catmaster_code:$PYTHONPATH
eval "$(conda shell.bash hook)"
conda activate catmaster-cpu
source /public/software/vasp.6.4.1-vtst-sol/env.sh
ulimit -s unlimited
export I_MPI_HYDRA_BOOTSTRAP=ssh