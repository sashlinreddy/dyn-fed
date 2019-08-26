#!/bin/bash
echo "Running slurm jobs for different number of workers"
workers_list=(8 32 64 128 216 100 200)
for value in "${workers_list[@]}"
do
    echo "Running job for $value workers"
	sbatch -n $value /home-mscluster/sreddy/fault-tolerant-ml/slurm_launch.sh -v 20
    echo "Completed for $value workers"
done

echo "COMPLETE! :)"