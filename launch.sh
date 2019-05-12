#!/bin/bash
#SBATCH -J ftml
#SBATCH -n 16
#SBATCH -o /home-mscluster/sreddy/logs/output.log
#SBATCH -t 10:00
#SBATCH -p batch

srun --multi-prog $HOME/tests/mult.conf
