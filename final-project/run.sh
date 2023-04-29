#!/bin/bash

# mpirun --bind-to none -mca btl ^openib --host a5 -npernode 1         \
#   numactl --physcpubind 0-63                                 \
#   ./main $@

mpirun --bind-to none -mca btl ^openib --host a0,a3,a4,a5 -npernode 1         \
  numactl --physcpubind 0-63                                 \
  ./main $@
