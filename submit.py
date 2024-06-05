#!/usr/bin/env python
#SBATCH --job-name=pretrain
#SBATCH --output=%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=16G
#SBATCH --qos=m3
#SBATCH --time=4:00:00
#SBATCH --signal=B:USR1@240
#SBATCH --open-mode=append

# -*- coding: utf-8 -*-

import os 

print(os.environ['SLURM_PROCID'])
print("Hello World!")