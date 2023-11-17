#!/bin/env bash

outfile=$(mktemp)
compute-sanitizer --log-file $outfile $1
rc=$?
output=$(<$outfile)
rm $outfile
if [[ $output =~ "Target application terminated before first instrumented API call" ]]; then
  exit 0
fi
exit $rc
