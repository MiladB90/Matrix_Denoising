#!/bin/sh

#SBATCH --job-name=md
#SBATCH --partition=normal,owners,donoho,hns,stat
#SBATCH --cpus-per-task=128
#SBATCH --time=24:30:00
#SBATCH --error=md_cs01.err
#SBATCH --output=md_cs01.out


## Run the python script
time python3 ./experiment.py
