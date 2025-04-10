#!/bin/bash

#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --job-name=quickpic
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
/home/myadav/prefix/bin/qpic_traj qpinput.json
