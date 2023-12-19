#!/bin/sh

#SBATCH --job-name=matrix_denoising
#SBATCH --partition=normal,owners,donoho,hns,stat
#SBATCH --cpus-per-task=30
#SBATCH --time=01:00:00
#SBATCH --error=md0001.err
#SBATCH --output=md0001.out


## Run the python script
time python3 ./experiment.py
