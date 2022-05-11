#!/bin/bash
# NSYS="nvprof --profile-child-processes "
EXEC_CMD="${NSYS} bazel run -c opt //benchmark/multi_commitment:benchmark -- "

############################################

verbose=1
spreadsheet_dir="benchmark/multi_commitment"
results_dir="benchmark/multi_commitment/.results"
curr_final_benchmark_file="${spreadsheet_dir}/spreadsheet.txt"

read_field() {
    # $1 --> column
    # $2 --> file 
    IFS=':'; read -a line_arr <<< $(awk '{if(NR=='${1}') print $0}' ${2}); 

    echo ${line_arr[1]}
}

results() {
    word_sizes=$1
    execs=$2
    backend=$3

    # Extract the duration time of each benchmark
    for ws in ${word_sizes[@]}; do
        for ex in ${execs[@]}; do
            c=$(echo $ex | cut -f1 -d-)
            r=$(echo $ex | cut -f2 -d-)
            curr_benchmark_file="${results_dir}/${ws}_${c}_${r}.${backend}.out"
            val=$(read_field 8 ${curr_benchmark_file})
            echo "$val"
        done
    done
}

run() {
    word_sizes=$1
    execs=$2
    backend=$3

    mkdir -p ${results_dir}

    # Run all the benchmarks
    for ws in ${word_sizes[@]}; do
        for ex in ${execs[@]}; do
            c=$(echo $ex | cut -f1 -d-)
            r=$(echo $ex | cut -f2 -d-)
            curr_benchmark_file="${results_dir}/${ws}_${c}_${r}.${backend}.out"
            ${EXEC_CMD} ${backend} ${c} ${r} ${ws} ${verbose} > ${curr_benchmark_file}
        done
    done

    results $word_sizes $execs $backend
}

valid_gpu() {
    word_sizes=$1
    execs=$2
    backend=$3

    # Extract the duration time of each benchmark
    for ws in ${word_sizes[@]}; do
        for ex in ${execs[@]}; do
            c=$(echo $ex | cut -f1 -d-)
            r=$(echo $ex | cut -f2 -d-)
            cpu_file="${results_dir}/${ws}_${c}_${r}.cpu.out"
            gpu_file="${results_dir}/${ws}_${c}_${r}.gpu.out"

            if [ -e "$cpu_file" ]; then
                if [ -e "$gpu_file" ]; then
                    diff ${cpu_file} ${gpu_file}
                fi
            fi
        done
    done
}

spreadsheet() {
    word_sizes=$1
    execs=$2
    backend=$3

    c0="Commitments\t"
    c1="Data Rows\t"
    c2="Word Size\t"
    c3="Table Size (MB)\t"
    c4="Total Exponentiations\t"
    c5="CPU Duration (s)\t"
    c6="CPU Throughput (Exponentiations / s)\t"
    c7="GPU Duration (s)\t"
    c8="GPU Throughput (Exponentiations / s)\t"

    cat /dev/null > ${curr_final_benchmark_file}
    echo -e $c0$c1$c2$c3$c4$c5$c6$c7$c8 >> ${curr_final_benchmark_file}

    # Extract the duration time of each benchmark
    for ws in ${word_sizes[@]}; do
        for ex in ${execs[@]}; do
            c=$(echo $ex | cut -f1 -d-)
            r=$(echo $ex | cut -f2 -d-)
            
            duration_cpu="-"
            throughput_cpu='-'
            duration_gpu="-"
            throughput_gpu='-'

            cpu_file="${results_dir}/${ws}_${c}_${r}.cpu.out"
            gpu_file="${results_dir}/${ws}_${c}_${r}.gpu.out"

            if [ -e "$cpu_file" ]; then
                duration_cpu=$(read_field 8 $cpu_file);
                throughput_cpu=$(read_field 9 $cpu_file);
            fi
            
            if [ -e "$gpu_file" ]; then
                table_size=$(read_field 5 $gpu_file);
                num_exps=$(read_field 6 $gpu_file);
                
                duration_gpu=$(read_field 8 $gpu_file);
                throughput_gpu=$(read_field 9 $gpu_file);
            fi

            echo -e "$c \t $r \t $ws \t $table_size \t $num_exps \t $duration_cpu \t $throughput_cpu \t $duration_gpu \t $throughput_gpu" >> ${curr_final_benchmark_file}
        done
    done
}

backend="$2"

if [ -z "$2" ]
  then
    backend="cpu"
fi

word_sizes=("4")

execs=()

num_commitments=("10" "100" "1000")
num_rows=("10" "100" "1000")

for c in ${num_commitments[@]}; do
    for r in ${num_rows[@]}; do
        execs+=("$c-$r")
    done
done

# edge cases : num_rows too big
# execs+=("1-1000000000")
execs+=("1-1000000" "1-10000000" "1-100000000" "1-1000000000")

# edge cases : num_commitments too big
execs+=("10000-1" "100000-1")

$1 $word_sizes $execs $backend # run specified function