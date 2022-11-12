#!/bin/bash

for d in {1..2}; do
    sbatch scripts/hmillclassifier.jl $d
done
