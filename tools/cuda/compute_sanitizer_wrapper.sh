#!/bin/env bash

outfile=$(mktemp)
compute-sanitizer --log-file $outfile $1
rc=$?
output=$(<$outfile)
rm $outfile

# compute-sanitizer will fail if a test doesn't launch any kernels which we don't
# want since we run it against all of our tests.
if [[ $output =~ "Target application terminated before first instrumented API call" ]]; then
  exit 0
fi
exit $rc
