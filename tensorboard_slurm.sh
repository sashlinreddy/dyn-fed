#!/bin/bash

#SBATCH -n 1
#SBATCH -t 04:00:00               # max runtime is 4 hours
#SBATCH -J tensorboard_server    # name
#SBATCH -o /home-mscluster/sreddy/logs/tensorflow/tb-%J.out
#SBATCH -p batch

# To run as an array job, use the following command:
# sbatch --partition=beards --array=0-0 tensorboardHam.sh
# squeue --user thpaul

let ipnport=6006
echo ipnport=$ipnport

ipnip=$(hostname -i)
echo ipnip=$ipnip

source /home-mscluster/sreddy/.bashrc
LOGDIR=/home-mscluster/sreddy/logs

conda activate ftml
tensorboard --logdir="${LOGDIR}" --port=$ipnport
