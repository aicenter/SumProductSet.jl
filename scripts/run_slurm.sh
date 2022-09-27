#!/bin/bash

for d in {1..20}; do
    sbatch scripts/real.jl $d
done
