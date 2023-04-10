#!/bin/bash

set -e

BENCH_TYPE=$1
BENCHMARK=$2
OUTPUT=$3
VALGRIND=valgrind
DOT=dot
GPROF2DOT=gprof2dot
BENCHMARK_NAME=$(basename $BENCHMARK)

rm -rf $OUTPUT

echo "#!/bin/bash" >> $OUTPUT
echo "set -e" >> $OUTPUT
echo "echo Cleaning callgrind data" >> $OUTPUT
echo "rm -rf *tar.gz *.zip *.pdf *.svg *.dot callgrind.out.[0-9]*" >> $OUTPUT
echo "BENCHMARK_NAME=\"$BENCHMARK_NAME\"" >> $OUTPUT

if [[ $BENCH_TYPE == "flamegraph" ]];
then
    echo "/root/.cargo/bin/flamegraph -o \"\$BENCHMARK_NAME-flamegraph.svg\" -- $BENCHMARK \"\${@:1}\"" >> $OUTPUT
    echo "rm -rf perf.data vgcore.*" >> $OUTPUT
else
    echo "$VALGRIND --tool=callgrind --collect-atstart=no $BENCHMARK \"\${@:1}\"" >> $OUTPUT
    echo "echo Converting benchmark result to image" >> $OUTPUT
    echo "$GPROF2DOT --format=callgrind --output=\$BENCHMARK_NAME.dot callgrind.out.[0-9]*" >> $OUTPUT
    echo "$DOT -Tsvg \$BENCHMARK_NAME.dot -o \"\$BENCHMARK_NAME-callgrind.svg\"" >> $OUTPUT
    echo "mv callgrind.out.[0-9]* callgrind.out.\$BENCHMARK_NAME" >> $OUTPUT
    echo "zip \$BENCHMARK_NAME.zip *.svg *.dot callgrind.out.*" >> $OUTPUT
    echo "tar zcf \$BENCHMARK_NAME.tar.gz *.svg *.dot callgrind.out.*" >> $OUTPUT
    echo "rm -rf \$BENCHMARK_NAME.dot callgrind.out.*" >> $OUTPUT
fi

chmod a+x $OUTPUT
