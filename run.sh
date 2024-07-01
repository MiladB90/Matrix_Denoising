#!/bin/sh

#SBATCH --job-name=md03
#SBATCH --partition=normal,owners,donoho,hns,stat
#SBATCH --cpus-per-task=32
#SBATCH --time=05:00:00
#SBATCH --error=md0003.err
#SBATCH --output=md0003.out


## Run the python script
time python3 ./experiment.py
