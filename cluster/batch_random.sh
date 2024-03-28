#!/bin/bash
#SBATCH -p kempner # partition (queue)
#SBATCH --account=kempner_bsabatini_lab
#SBATCH --job-name=eplhb-models     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --gres=gpu:1             # number of allocated gpus per node
#SBATCH --cpus-per-task=16        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=250G                # total memory per node (16 GB per cpu-core is default)
#SBATCH --time=2-02:00:00          # total run time limit (HH:MM:SS)
#SBATCH -o %j.out # STDOUT
#SBATCH -e %j.err # STDERR
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --mail-user shunli@g.harvard.edu

module load python/3.10.12-fasrc01
python -c 'print("Python 3.10.12 loaded...")'
mamba activate eplhbmodel
python -c 'print("Environment activated...")'

# Make directory of today's date
today=$(date +%Y%m%d)
mkdir -p /n/holylabs/LABS/bsabatini_lab/Users/shunnnli/EP-LHb-model/results/Random/$today
python -c 'print("Created experiment directory at lab folder: EP-LHb-model/results/Random/'$today'")'

python -c 'print("************ Batch experiments ************")'
# srun python ~/code/EP-LHb-model/models_Random.py
srun python ~/code/EP-LHb-model/rnn_sparsity_random.py
python -c 'print("************ Batch experiments ************")'

# Move job outputs to experiment directory
mv $SLURM_JOB_ID.out /n/holylabs/LABS/bsabatini_lab/Users/shunnnli/EP-LHb-model/results/Random/$today
mv $SLURM_JOB_ID.err /n/holylabs/LABS/bsabatini_lab/Users/shunnnli/EP-LHb-model/results/Random/$today
python -c 'print("Moved job outputs to experiment directory")'