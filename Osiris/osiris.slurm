#!/bin/bash

#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --job-name=osiris
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
mpirun -np 64 --oversubscribe /home/kylaletko/osiris/run/osiris-3D.e input
