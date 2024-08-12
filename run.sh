#!/bin/sh

#SBATCH --job-name=md
#SBATCH --partition=normal,owners,donoho,hns,stat
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00
#SBATCH --error=md_test.err
#SBATCH --output=md_test.out


## Run the python script
time python3 ./experiment.py
