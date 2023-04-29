#!/bin/bash

NSYS=/usr/local/cuda-11.7/bin/nsys

${NSYS} --version

# ${NSYS} profile --force-overwrite=true \
#                 --trace=cuda,nvtx --stats=true \
#                 --output /home/n1/junyeol/hpc/final-project/profiles/N \
#                 numactl --physcpubind 0-63 ./main "$@"

${NSYS} profile --force-overwrite=true \
                --trace=cuda,nvtx --stats=true \
                --output /home/n1/junyeol/hpc/matmul-opt/profiles/A \
                numactl --physcpubind 0-63 ./main "$@"