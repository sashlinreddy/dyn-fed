#!/bin/bash
NWORKERS=$1
echo "No of workers: $NWORKERS"

while [ "$2" != "" ]; do
    case $2 in
        -v| --VERBOSE )         shift
                                VERBOSE="-v $2"
                                ;;
        -m| --MODEL )         shift
                                MODEL="$2"
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
    pythonw examples/train_logistic.py $NWORKERS -r master $VERBOSE
elif [ "$MODEL" == "LOG2" ]; then
    echo "Running Logistic Model"
    pythonw examples/train_logisticv2.py $NWORKERS -r master $VERBOSE
elif [ "$MODEL" == "NN" ]; then
    echo "Running NN Model"
    pythonw examples/train_nn.py $NWORKERS -r master -$VERBOSE
fi