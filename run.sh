#!/bin/sh

#SBATCH --job-name=matrix_denoising
#SBATCH --partition=normal,owners,donoho,hns,stat
#SBATCH --cpus-per-task=30
#SBATCH --time=00:30:00
#SBATCH --error=md0002-2.err
#SBATCH --output=md0002-2.out


## Run the python script
time python3 ./experiment.py
