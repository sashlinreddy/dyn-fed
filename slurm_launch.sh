#!/bin/bash
#SBATCH -J ftml
#SBATCH -o /home-mscluster/sreddy/logs/slurm/slurm_%j.log
#SBATCH -t 10:00
#SBATCH -p batch
#SBATCH --export=LOGDIR=/home-mscluster/sreddy/logs/slurm,TFDIR=/home-mscluster/sreddy/logs/,FIGDIR=/home-mscluster/sreddy/fault-tolerant-ml/reports/figures

while [ "$1" != "" ]; do
    case $1 in
        -i | --n_iterations )   shift
                                n_iterations="-i $1"
                                ;;
        -lr | --learning_rate)  shift
                                learning_rate="-lr $1"
                                ;;
        -v| --verbose )         shift
                                verbose="-v $1"
                                ;;
        -s | --scenario)        shift
                                scenario="-s $1"
                                ;;
        -r | --remap )          shift
                                remap="-r $1"
                                ;;
        -q | --quantize)        shift
                                quantize="-q $1"
                                ;;
        -nmp | --n_most_rep )   shift
                                n_most_rep="-nmp $1"
                                ;;
        -cp | --comm_period)    shift
                                comm_period="-cp $1"
                                ;;
        -t | --timeout )        shift
                                timeout="-t $1"
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

echo -e '0\t' $PYTHON_EXE $MASTER_EXE $DATA_DIR $SLURM_NTASKS $n_iterations $learning_rate \
$verbose $scenario $remap $quantize $n_most_rep $comm_period $timeout \
> /home-mscluster/sreddy/fault-tolerant-ml/m_w_$SLURM_JOBID.conf
echo -e '*\t' $PYTHON_EXE $WORKER_EXE $DATA_DIR -i %t >> /home-mscluster/sreddy/fault-tolerant-ml/m_w_$SLURM_JOBID.conf

srun --multi-prog /home-mscluster/sreddy/fault-tolerant-ml/m_w_$SLURM_JOBID.conf
