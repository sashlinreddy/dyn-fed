#!/bin/bash
#SBATCH -J ftml
#SBATCH -o /home-mscluster/sreddy/logs/slurm_%j.log
#SBATCH -t 10:00
#SBATCH -p batch
#SBATCH --export=LOGDIR=/home-mscluster/sreddy/logs

export PYTHON_EXE=/home-mscluster/sreddy/miniconda3/envs/ftml/bin/python3
export WORKER_EXE=/home-mscluster/sreddy/dyn-fed/tests/run_worker.py
export DATA_DIR=/home-mscluster/sreddy/dyn-fed/data

echo -e '*\t' $PYTHON_EXE $WORKER_EXE $DATA_DIR -i %t -a 1 >> /home-mscluster/sreddy/dyn-fed/m_w_$SLURM_JOBID.conf

srun --multi-prog /home-mscluster/sreddy/dyn-fed/m_w_$SLURM_JOBID.conf

