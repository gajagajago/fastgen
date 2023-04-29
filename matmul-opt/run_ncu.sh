#!/bin/bash

KERNEL=2d_blocktiling

sudo /usr/local/cuda-11.7/bin/ncu \
    --set full \
    --target-processes all \
    -o /home/n1/junyeol/hpc/matmul-opt/profiles/${KERNEL} \
    numactl --physcpubind 0-63 ./main "$@"