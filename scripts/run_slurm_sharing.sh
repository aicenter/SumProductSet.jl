#!/bin/bash

for d in {1..8}; do
    sbatch scripts/nodesharing.jl $d
done
