#!/bin/bash

WORLD_RANK=${OMPI_COMM_WORLD_RANK}
NSYS=/usr/local/cuda-11.7/bin/nsys

${NSYS} --version

# ${NSYS} profile --force-overwrite=true \
#                 --trace=cuda,nvtx --stats=true \
#                 --output /home/n1/junyeol/hpc/final-project/profiles/N \
#                 numactl --physcpubind 0-63 ./main "$@"

${NSYS} profile --force-overwrite=true \
                --trace=cuda,nvtx,mpi --stats=true \
                --output /home/n1/junyeol/hpc/final-project/profiles/E${WORLD_RANK} \
                numactl --physcpubind 0-63 ./main "$@"