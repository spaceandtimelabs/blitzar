#!/bin/bash

set -e

BENCHMARK=$1
OUTPUT=$2
VALGRIND=valgrind
DOT=dot
GPROF2DOT=gprof2dot
BENCHMARK_NAME=$(basename $BENCHMARK)

rm -rf $OUTPUT

echo "#!/bin/bash" >> $OUTPUT
echo "set -e" >> $OUTPUT
echo "echo Cleaning callgrind data" >> $OUTPUT
echo "rm -rf *tar.gz *.zip \$BENCHMARK_NAME.pdf \$BENCHMARK_NAME.svg \$BENCHMARK_NAME.dot callgrind.out.[0-9]*" >> $OUTPUT
echo "BENCHMARK_NAME=\"$BENCHMARK_NAME\$(printf _\"%s\" \"\${@:1}\")\"" >> $OUTPUT
echo "$VALGRIND --tool=callgrind $BENCHMARK \"\${@:1}\"" >> $OUTPUT
echo "echo Converting benchmark result to image" >> $OUTPUT
echo "$GPROF2DOT --format=callgrind --output=\$BENCHMARK_NAME.dot callgrind.out.[0-9]*" >> $OUTPUT
echo "$DOT -Tsvg \$BENCHMARK_NAME.dot -o \"\$BENCHMARK_NAME.svg\"" >> $OUTPUT
echo "mv callgrind.out.[0-9]* callgrind.out.\$BENCHMARK_NAME" >> $OUTPUT
echo "zip \$BENCHMARK_NAME.zip *.svg *.dot callgrind.out.*" >> $OUTPUT
echo "tar zcf \$BENCHMARK_NAME.tar.gz *.svg *.dot callgrind.out.*" >> $OUTPUT
echo "rm -rf \$BENCHMARK_NAME.dot callgrind.out.*" >> $OUTPUT

chmod a+x $OUTPUT
