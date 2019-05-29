#!/bin/bash
#SBATCH -J ftml
#SBATCH -n 20
#SBATCH -o $HOME/logs/output.log
#SBATCH -t 10:00
#SBATCH -p batch
#SBATCH --export=LOGDIR=$HOME/logs

srun --multi-prog ./m_w.conf
