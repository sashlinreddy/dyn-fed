#!/bin/bash
# Script deprecated

# n_workers=6 # Default no. of workers
# if [ $# -gt 0 ]; then
# 	if [ $1 == "-n" ]; then
# 		shift
# 		n_workers=$1
# 	fi
# fi

#!/bin/bash
echo "Running slurm jobs for different number of workers"
workers_list=( 10 20 26 39 48)
for value in "${workers_list[@]}"
do
	echo $value
done
