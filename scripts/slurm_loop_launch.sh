#!/bin/bash
echo "Running slurm jobs for different number of clients"
workers_list=(8 16 32 64)
for value in "${workers_list[@]}"
do
    echo "Running job for $value clients"
	sbatch -n $value /home-mscluster/sreddy/dyn-fed/scripts/slurm_launch.sh -v INFO
    echo "Completed for $value clients"
done

echo "COMPLETE! :)"