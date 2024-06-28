#!/bin/bash
#######################################################
# Job Script for Submission of Inno4Scale Simulations #
#######################################################

# -------------- Slurm Settings -----------------

#SBATCH --job-name=inno4scale001
#SBATCH --mail-type=END
#SBATCH --mail-user=maximilian.kruse@kit.edu
#SBATCH --partition=single
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=10GB
#SBATCH --time=05:00:00
#SBATCH --output=%x_%A-%a.log
#SBATCH --error=%x_%A-%a.err


# ------------ Command Execution ----------------

pyenv activate inno4scale
python run.py -app dummy