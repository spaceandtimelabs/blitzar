#!/bin/bash

############################################

results_dir="benchmark/multi_commitment/.results/"

word_sizes=("4")

cols=("10" "100" "1000")
rows=("10" "100" "1000")

# Extract the duration time of each benchmark
for ws in ${word_sizes[@]}; do
    echo word_size=${ws}

    for c in ${cols[@]}; do
        for r in ${rows[@]}; do
            cpu_file="${results_dir}/${ws}_${c}_${r}.cpu.out"
            gpu_file="${results_dir}/${ws}_${c}_${r}.gpu.out"

            diff ${cpu_file} ${gpu_file}
        done
    done
done