#!/bin/bash

set -e

i=1
for dir in */; do
    abs_dir=$(readlink -f "$dir")
    echo "$abs_dir"
    grep WallClockTime "$abs_dir"/*.o* | awk '{print $NF}' > amr-wind-time-$i.txt
   $abs_dir/../../amr-wind-average.py -f amr-wind-time-$i.txt > amr-wind-avg-$i.txt
   rm amr-wind-time-$i.txt
    ((i=i+1))
done
