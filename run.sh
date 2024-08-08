#!/bin/sh

#SBATCH --job-name=md06
#SBATCH --partition=normal,owners,donoho,hns,stat
#SBATCH --cpus-per-task=64
#SBATCH --time=5:00:00
#SBATCH --error=md06.err
#SBATCH --output=md06.out


## Run the python script
time python3 ./experiment.py
