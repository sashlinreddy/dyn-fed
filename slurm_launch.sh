#!/bin/bash
#SBATCH -J ftml
#SBATCH -o /home-mscluster/sreddy/logs/slurm/slurm_%j.log
#SBATCH -t 10:00
#SBATCH -p batch
#SBATCH --export=LOGDIR=/home-mscluster/sreddy/logs/slurm,TFDIR=/home-mscluster/sreddy/logs/,FIGDIR=/home-mscluster/sreddy/fault-tolerant-ml/reports/figures,PROJECT_DIR=/home-mscluster/sreddy/fault-tolerant-ml

while [ "$1" != "" ]; do
    case $1 in
        -v| --verbose )         shift
                                verbose="-v $1"
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done

export PYTHON_EXE=/home-mscluster/sreddy/miniconda3/envs/ftml/bin/python3
export MASTER_EXE=/home-mscluster/sreddy/fault-tolerant-ml/tests/run_master.py
export WORKER_EXE=/home-mscluster/sreddy/fault-tolerant-ml/tests/run_worker.py
export DATA_DIR=/home-mscluster/sreddy/fault-tolerant-ml/data

echo -e '0\t' $PYTHON_EXE $MASTER_EXE $SLURM_NTASKS $verbose > /home-mscluster/sreddy/fault-tolerant-ml/m_w_$SLURM_JOBID.conf
echo -e '*\t' $PYTHON_EXE $WORKER_EXE $SLURM_NTASKS $verbose -i %t >> /home-mscluster/sreddy/fault-tolerant-ml/m_w_$SLURM_JOBID.conf

srun --multi-prog /home-mscluster/sreddy/fault-tolerant-ml/m_w_$SLURM_JOBID.conf
