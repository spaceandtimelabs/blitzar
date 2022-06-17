#!/bin/bash
# NSYS="nvprof --profile-child-processes "
EXEC_CMD="${NSYS} bazel run -c opt //benchmark/multi_commitment:benchmark -- "

############################################

verbose=0
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

validate_gpu() {
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
    c5="Duration (s)\t"
    c6="Std. Deviation (s)\t"
    c7="Throughput (Exponentiations / s)\t"

    c8="Duration (s)\t"
    c9="Std. Deviation (s)\t"
    c10="Throughput (Exponentiations / s)\t"

    cat /dev/null > ${curr_final_benchmark_file}
    echo -e $c0$c1$c2$c3$c4$c5$c6$c7$c8$c9$c10 >> ${curr_final_benchmark_file}

    # Extract the duration time of each benchmark
    for ws in ${word_sizes[@]}; do
        for ex in ${execs[@]}; do
            c=$(echo $ex | cut -f1 -d-)
            r=$(echo $ex | cut -f2 -d-)

            bench_file="${results_dir}/${ws}_${c}_${r}.${backend}.out"
            
            if [ -e "$bench_file" ]; then
                table_size=$(read_field 6 $bench_file);
                num_exps=$(read_field 7 $bench_file);
                
                with_gens_duration=$(read_field 9 $bench_file);
                with_gens_deviation=$(read_field 10 $bench_file);
                with_gens_throughput=$(read_field 11 $bench_file);

                no_gens_duration=$(read_field 13 $bench_file);
                no_gens_deviation=$(read_field 14 $bench_file);
                no_gens_throughput=$(read_field 15 $bench_file);
            fi

            echo -e "$c \t $r \t $ws \t $table_size \t $num_exps \t $no_gens_duration \t $no_gens_deviation \t $no_gens_throughput \t $with_gens_duration \t $with_gens_deviation \t $with_gens_throughput" >> ${curr_final_benchmark_file}
        done
    done
}

backend="$2"

if [ -z "$2" ]
  then
    backend="cpu"
fi

word_sizes=("32")

execs=()

# num_rows=("10" "100" "1000")
# num_commitments=("10" "100" "1000")

for c in ${num_commitments[@]}; do
    for r in ${num_rows[@]}; do
        execs+=("$c-$r")
    done
done

# edge cases : num_rows too big
execs+=("1-1" "1-10" "1-100" "1-1000" "1-10000" "1-100000")

# edge cases : num_commitments too big
execs+=("10-1" "100-1" "1000-1" "10000-1")

$1 $word_sizes $execs $backend # run specified function
