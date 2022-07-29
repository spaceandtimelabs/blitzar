#!/bin/bash
EXEC_CMD="bazel run -c opt //benchmark/multiprod1:benchmark -- "
EXEC_CMD_CALLGRIND="bazel run -c opt --define callgrind=true //benchmark/multiprod1:benchmark_callgrind -- "

results_dir="$(pwd)/benchmark/multiprod1/.results"

run() {
    use_callgrind=$1
    use_naive=$2
    sequence_length=$3
    num_sequences=$4
    max_num_inputs=$5
    num_samples=$6
    verbose=1

    base_dir="${results_dir}/use_naive_${use_naive}/"

    mkdir -p ${base_dir}

    curr_benchmark_file="${base_dir}/out"

    input_params="$use_naive $sequence_length $num_sequences $max_num_inputs $num_samples $verbose"

    if [ "${use_callgrind}" == "2" ]; then
        $EXEC_CMD_CALLGRIND $input_params
        mv -f *.svg ${curr_benchmark_file}.svg
    else
        $EXEC_CMD $input_params > ${curr_benchmark_file}.out
    fi
}

use_naive=$1
num_samples=50
max_num_inputs=4000
num_sequences=30
sequence_length=1000

run 1 $use_naive $sequence_length $num_sequences $max_num_inputs $num_samples

run 2 $use_naive $sequence_length $num_sequences $max_num_inputs 1
