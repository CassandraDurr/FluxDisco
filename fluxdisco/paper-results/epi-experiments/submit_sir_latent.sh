#!/bin/bash

#SBATCH -J sir_latent
#SBATCH -c 73
#SBATCH --mem=100G
#SBATCH --mail-user=c.durr@lancaster.ac.uk
#SBATCH --mail-type=ALL

# Output and error files
#SBATCH -o out/job_%j.out
#SBATCH -e out/job_%j.err

# Create out
mkdir -p out

# Init conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate py314

# git pull
python run_sir_latent.py --system sir_prevalence_only_1 sir_incidence_only_1 sir_both_1 --regime Standard
