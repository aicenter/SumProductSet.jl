#!/bin/bash

ndatasets=16
ngrid=150
max_jobs=400
start=1

export start
export max_jobs
export ngrid
export ndatasets

function run() {
    read unavail <<< $(
    squeue --user=$USER "$@" | awk '
    BEGIN {
        abbrev["R"]="(Running)"
        abbrev["PD"]="(Pending)"
        abbrev["CG"]="(Completing)"
        abbrev["F"]="(Failed)"
        unvail=0
    }
    NR>1 {a[$5]++}
    END {
        for (i in a) {
            # printf "%-2s %-12s %d\n", i, abbrev[i], a[i]
            unavail=unavail+a[i]
        }
        printf "%d\n", unavail
    }'
    )
    echo "Unvailable slots:" $unavail;

    avail=$(($max_jobs-$unavail))
    if [ $avail -gt 0 ]; then
        end=$(($start + $avail))
    else
        end=0
    fi 
    echo "Available slots:" $avail;

    counter=1
    for d in $(seq 1 $ndatasets); do
        for c in $(seq 1 $ngrid); do
            if (( $counter > $start && $counter <= $end )); then
                sbatch scripts/run_job.sh $c $d
                echo "Submitted job with dataset:" $d "and cofig ID:" $c
            fi
            counter=$(($counter+1))

            if [ $counter -gt $end ]; then
                start=$end
                export start
            fi
        done
    done
}

export -f run
# run in 2 minute intervals
watch -n 120 run -r
