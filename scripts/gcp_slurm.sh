#!/bin/bash
#SBATCH -J ftml
#SBATCH -o /home/g675723_students_wits_ac_za/logs/slurm/slurm_%j.log
#SBATCH -t 10:00
#SBATCH -p debug
#SBATCH --export=LOGDIR=/home/g675723_students_wits_ac_za/logs/slurm,FIGDIR=/home/g675723_students_wits_ac_za/fault-tolerant-ml/reports/figures,PROJECT_DIR=/home/g675723_students_wits_ac_za/fault-tolerant-ml
# ,TFDIR=/home/g675723_students_wits_ac_za/logs/

while [ "$1" != "" ]; do
    case $1 in
        -v| --verbose )         shift
                                verbose="-v $1"
                                ;;
        -m| --MODEL )           shift
                                MODEL="$1"
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done

# Decide on the type of model
EXE_PATH=/home/g675723_students_wits_ac_za/fault-tolerant-ml/examples/train_logistic.py

if [ "$MODEL" == "LINEAR" ]; then
    echo "Running Linear Model"
    EXE_PATH=/home/g675723_students_wits_ac_za/fault-tolerant-ml/examples/train_linear.py
elif [ "$MODEL" == "LOG2" ]; then
    echo "Running Logistic 2 Model"
    EXE_PATH=/home/g675723_students_wits_ac_za/fault-tolerant-ml/examples/train_logisticv2.py
elif [ "$MODEL" == "NN" ]; then
    echo "Running NN Model"
    EXE_PATH=/home/g675723_students_wits_ac_za/fault-tolerant-ml/examples/train_nn.py
fi

export PYTHON_EXE=/home/g675723_students_wits_ac_za/miniconda3/envs/ftml/bin/python3
export MASTER_EXE=$EXE_PATH
export WORKER_EXE=$EXE_PATH
export DATA_DIR=/home/g675723_students_wits_ac_za/fault-tolerant-ml/data/fashion-mnist

echo -e '0\t' $PYTHON_EXE $MASTER_EXE $SLURM_NTASKS -r master $verbose > /home/g675723_students_wits_ac_za/fault-tolerant-ml/m_w_$SLURM_JOBID.conf
echo -e '*\t' $PYTHON_EXE $WORKER_EXE $SLURM_NTASKS -r worker $verbose -i %t >> /home/g675723_students_wits_ac_za/fault-tolerant-ml/m_w_$SLURM_JOBID.conf

srun --multi-prog /home/g675723_students_wits_ac_za/fault-tolerant-ml/m_w_$SLURM_JOBID.conf
