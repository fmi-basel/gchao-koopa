#!/bin/sh

#SBATCH --mail-user=bastian.eichenberger@fmi.ch
#SBATCH --mail-type=END
#SBATCH --account=fly-image
#SBATCH --job-name="SMonster" # Something that helps you recognize your job in the queue
#SBATCH --output=../log_luigi/%x-%j.out
#SBATCH -N 1 # ensure that all cores are on one machine
#SBATCH --ntasks=1    # non MPI applications are usually single task, but you can do multiple tasks as well
#SBATCH --time=02:00:00    # time required by the job, if you hit this limit the job will be terminated
#SBATCH --cpus-per-task=48   # how many CPU cores per task do you need?
#SBATCH --partition=cpu_short # for CPU only change this, e.g. to cpu_short, see also sinfo command output

# Configuration
echo "Starting"

CONDA_DIR=/tungstenfs/scratch/gchao/eichbast/miniconda/bin/activate
source $CONDA_DIR image
echo "Activated conda"

# Run
python -m luigi --module merge Merge --workers=24 --local-scheduler
