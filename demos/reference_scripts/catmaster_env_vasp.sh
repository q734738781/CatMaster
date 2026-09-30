export PATH=<CONDA_BIN>:<VASP_BIN>:$PATH
export PYTHONPATH=<CATMASTER_REPO_OR_INSTALL_PATH>:$PYTHONPATH
eval "$(conda shell.bash hook)"
conda activate <CATMASTER_CPU_ENV>
source <VASP_ENV_SCRIPT>
ulimit -s unlimited
export I_MPI_HYDRA_BOOTSTRAP=ssh
