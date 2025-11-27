The Phase1 of this project aims to execute real VASP program in the specified CPU/GPU cluster using jobflow-remote v0.1.8. For both servers, it should support Slurm job submit or Run locally on the server, and support login with private key or password, if the server do not support private key login.

For this scenario, an example slurm script for VASP calculation is provided in templates/task_vasp.sh

For the first stage, two instances should be considered as testing demos. One is using VASP for relaxing O2 molecule in the box, where a POSCAR Geometry file is in demo_scripts/assets/POSCAR_O2_BOX, and you can use MPRelaxSet and modify ISIF=2 for the following VASP settings (Of course, K-point 1x1x1). Second is using MACE on the gpu server for MACE relaxation of the same instance.

Currently, jobflow-remote and pymatgen, ase, etc common packages is installed in the pc/cpu/gpu, and MACE torch is installed specially in GPU server. The connection information is presented in configs/server_connection_information.yaml.

