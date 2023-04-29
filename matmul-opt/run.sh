#!/bin/bash

numactl --physcpubind 0-63                                 \
  ./main $@