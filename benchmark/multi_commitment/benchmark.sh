#!/bin/bash
USE_CALLGRIND="0"
EXEC_CMD="bazel run -c opt //benchmark/multi_commitment:benchmark -- "
EXEC_CMD_CALLGRIND="bazel run -c opt //benchmark/multi_commitment:benchmark_callgrind -- "

############################################

verbose=1
num_samples=3
spreadsheet_dir="benchmark/multi_commitment"
results_dir="benchmark/multi_commitment/.results"
curr_final_benchmark_file="${spreadsheet_dir}/spreadsheet.txt"

read_field() {
    # $1 --> column
    # $2 --> file 
    IFS=':'; read -a line_arr <<< $(awk '{if(NR=='${1}') print $0}' ${2}); 

    echo ${line_arr[1]}
}

run() {
    word_sizes=$1
    execs=$2
    backend=$3

    # Run all the benchmarks
    for ws in ${word_sizes[@]}; do
        for ex in ${execs[@]}; do
            c=$(echo $ex | cut -f1 -d-)
            r=$(echo $ex | cut -f2 -d-)

            base_dir="${results_dir}/${c}_commits/${ws}_wordsize/"

            mkdir -p ${base_dir}

            curr_benchmark_file="${base_dir}/${backend}_${c}_commits_${r}_rows_${ws}_wordsize_${num_samples}_samples.out"
            ${EXEC_CMD} ${backend} ${c} ${r} ${ws} ${verbose} ${num_samples} > ${curr_benchmark_file}
            
            if [ "${USE_CALLGRIND}" == "1" ]; then
                ${EXEC_CMD_CALLGRIND} ${backend} ${c} ${r} ${ws} ${verbose} ${num_samples}
                mv -f *.svg ${base_dir}/
            fi
        done
    done
}

spreadsheet() {
    word_sizes=$1
    execs=$2
    backend=$3

    c0="Commitments\t"
    c1="Commitment Length\t"
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

            bench_file="${results_dir}/${backend}_${c}_${r}_${ws}_${verbose}_${num_samples}.out"
            
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
USE_CALLGRIND="$3"

word_sizes=("0" "1" "4" "8" "32")

execs=()

# execs+=("1-10")

############################################
# Number of Commitments - Small
############################################

NC="1"
execs+=(
    "${NC}-1" "${NC}-10" "${NC}-50" "${NC}-100" "${NC}-250"
    "${NC}-500" "${NC}-1000" "${NC}-2000" "${NC}-5000"
    "${NC}-10000" "${NC}-100000"
)

NC="2"
execs+=(
    "${NC}-1" "${NC}-10" "${NC}-50" "${NC}-100" "${NC}-250"
    "${NC}-500" "${NC}-1000" "${NC}-2000" "${NC}-5000"
    "${NC}-10000" "${NC}-100000"
)

NC="5"
execs+=(
    "${NC}-1" "${NC}-10" "${NC}-50" "${NC}-100" "${NC}-250"
    "${NC}-500" "${NC}-1000" "${NC}-2000" "${NC}-5000"
    "${NC}-10000" "${NC}-100000"
)

############################################
# Number of Commitments - Medium
############################################

NC="10"
execs+=(
    "${NC}-1" "${NC}-10" "${NC}-50" "${NC}-100" "${NC}-250"
    "${NC}-500" "${NC}-1000" "${NC}-2000" "${NC}-5000" "${NC}-10000"
)

NC="50"
execs+=(
    "${NC}-1" "${NC}-10" "${NC}-50" "${NC}-100" "${NC}-250"
    "${NC}-500" "${NC}-1000" "${NC}-2000" "${NC}-5000" "${NC}-10000"
)

NC="100"
execs+=(
    "${NC}-1" "${NC}-10" "${NC}-50" "${NC}-100" "${NC}-250"
    "${NC}-500" "${NC}-1000" "${NC}-2000" "${NC}-5000"
)

############################################
# Number of Commitments - Large
############################################

NC="250"
execs+=(
    "${NC}-1" "${NC}-10" "${NC}-50" "${NC}-100" "${NC}-250"
    "${NC}-500" "${NC}-1000" "${NC}-2000"
)

NC="500"
execs+=(
    "${NC}-1" "${NC}-10" "${NC}-50" "${NC}-100" "${NC}-250"
    "${NC}-500" "${NC}-1000"
)

NC="1000"
execs+=(
    "${NC}-1" "${NC}-10" "${NC}-50" "${NC}-100" "${NC}-250"
    "${NC}-500"
)

$1 $word_sizes $execs $backend # run specified function
