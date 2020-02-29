#!/bin/bash
echo "Generate proto bindings"
protoc -I=protos/ --python_out=dyn_fed/proto/ protos/ftml.proto