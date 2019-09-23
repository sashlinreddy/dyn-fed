#!/bin/bash

while [ "$1" != "" ]; do
    case $1 in
        -n| --NWORKERS )         shift
                                NWORKERS="$1"
                                ;;
        -v| --VERBOSE )         shift
                                VERBOSE="-v $1"
                                ;;
        -m| --MODEL )         shift
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
if [ "$MODEL" == "LINEAR" ]; then
    echo "Running Linear Model"
    pythonw examples/train_linear.py $NWORKERS -r master $VERBOSE
elif [ "$MODEL" == "LOGISTIC" ]; then
    echo "Running Logistic Model"
    pythonw examples/train_logistic.py 8 -r master -v INFO
elif [ "$MODEL" == "NN" ]; then
    echo "Running NN Model"
    pythonw examples/train_nn.py 8 -r master -v INFO
fi