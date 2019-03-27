#!/bin/bash
# Basic range in for loop
echo Starting subscribers
for value in {1..10}
do
	python fault_tolerant_ml/worker.py &
done
echo Starting publisher
python fault_tolerant_ml/master.py
