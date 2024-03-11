#!/bin/bash
#SBATCH -p kempner_requeue # partition (queue)
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --job-name=eplhb-models     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --gres=gpu:1             # number of allocated gpus per node
#SBATCH --cpus-per-task=16        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=250G                # total memory per node (16 GB per cpu-core is default)
#SBATCH --time=00:30:00          # total run time limit (HH:MM:SS)
#SBATCH -o /jobs/EP-LHb-model/%j.out # STDOUT
#SBATCH -e /jobs/EP-LHb-model/%j.err # STDERR
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user shun.li@g.harvard.edu

module load python/3.10.12-fasrc01
mamba activate eplhbmodel

python -c 'print("Batch experiments started...")'
srun python ~/code/EP-LHb-model/models_MNIST.py