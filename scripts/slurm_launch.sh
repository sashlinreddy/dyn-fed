#!/bin/bash
#SBATCH -J ftml
#SBATCH -o /home-mscluster/sreddy/logs/slurm/slurm_%j.log
#SBATCH -t 07:00
#SBATCH -p batch
#SBATCH --export=LOGDIR=/home-mscluster/sreddy/logs/slurm,FIGDIR=/home-mscluster/sreddy/dyn-fed/reports/figures,PROJECT_DIR=/home-mscluster/sreddy/dyn-fed
# ,TFDIR=/home-mscluster/sreddy/logs/

while [ "$1" != "" ]; do
    case $1 in
        -v| --verbose )         shift
                                verbose="-v $1"
                                ;;
        -m| --MODEL )           shift
                                MODEL="$1"
                                ;;
        -c| --config )          shift
                                config="-c $1"
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
EXE_PATH=/home-mscluster/sreddy/dyn-fed/examples/train_logistic.py

if [ "$MODEL" == "LINEAR" ]; then
    echo "Running Linear Model"
    EXE_PATH=/home-mscluster/sreddy/dyn-fed/examples/train_linear.py
elif [ "$MODEL" == "V2" ]; then
    echo "Running Version2 Model"
    EXE_PATH=/home-mscluster/sreddy/dyn-fed/examples/train_model.py
elif [ "$MODEL" == "NN" ]; then
    echo "Running NN Model"
    EXE_PATH=/home-mscluster/sreddy/dyn-fed/examples/train_nn.py
elif [ "$MODEL" == "TF" ]; then
    echo "Running TF Model"
    EXE_PATH=/home-mscluster/sreddy/dyn-fed/examples/train_tf_model.py
fi

export PYTHON_EXE=/home-mscluster/sreddy/miniconda3/envs/ftml/bin/python3
export MASTER_EXE=$EXE_PATH
export WORKER_EXE=$EXE_PATH

echo -e '0\t' $PYTHON_EXE $MASTER_EXE $SLURM_NTASKS -r server $verbose $config > /home-mscluster/sreddy/dyn-fed/m_w_$SLURM_JOBID.conf
echo -e '*\t' $PYTHON_EXE $WORKER_EXE $SLURM_NTASKS -r client $verbose -i %t $config >> /home-mscluster/sreddy/dyn-fed/m_w_$SLURM_JOBID.conf

srun --multi-prog /home-mscluster/sreddy/dyn-fed/m_w_$SLURM_JOBID.conf
