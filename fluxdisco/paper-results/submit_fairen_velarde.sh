#!/bin/bash

#SBATCH -J fairen-velarde
#SBATCH -c 25
#SBATCH --mem=200G
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
python run_paper_experiments.py --systems fairen-velarde
