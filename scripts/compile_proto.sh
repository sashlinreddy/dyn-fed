#!/bin/bash
echo "Generate proto bindings"
protoc -I=protos/ --python_out=fault_tolerant_ml/proto/ protos/ftml.proto