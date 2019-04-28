#!/bin/bash
n_workers=6 # Default no. of workers
if [ $# -gt 0 ]; then
	if [ $1 == "-n" ]; then
		shift
		n_workers=$1
	fi
fi

# Basic range in for loop
echo Starting subscribers
for value in {1..$n_workers} 
do
	python fault_tolerant_ml/worker.py &
done
echo Starting publisher

python fault_tolerant_ml/master.py
