#!/bin/bash
echo "Running slurm jobs for different number of workers"
workers_list=(8 16 32 64)
for value in "${workers_list[@]}"
do
    echo "Running job for $value workers"
	sbatch -n $value /home-mscluster/sreddy/fault-tolerant-ml/scripts/slurm_launch.sh -v INFO
    echo "Completed for $value workers"
done

echo "COMPLETE! :)"