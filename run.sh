#!/bin/sh

#SBATCH --job-name=md07
#SBATCH --partition=normal,owners,donoho,hns,stat
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --error=md07.err
#SBATCH --output=md07.out


## Run the python script
time python3 ./experiment.py
