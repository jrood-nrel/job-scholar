#!/bin/bash

set -e

i=1
for dir in */; do
    abs_dir=$(readlink -f "$dir")
    echo "$abs_dir"
    grep Exawind::Total "$abs_dir"/*.o* > exawind-time-$i.txt
    grep Nalu-Wind::Total "$abs_dir"/*.o* > nalu-wind-time-$i.txt
    grep AMR-Wind::Total "$abs_dir"/*.o* > amr-wind-time-$i.txt
    grep Tioga::Total "$abs_dir"/*.o* > tioga-time-$i.txt
   ./average.py -f exawind-time-$i.txt > exawind-avg-$i.txt
   ./average.py -f nalu-wind-time-$i.txt > nalu-wind-avg-$i.txt
   ./average.py -f amr-wind-time-$i.txt > amr-wind-avg-$i.txt
   ./average.py -f tioga-time-$i.txt > tioga-avg-$i.txt
   rm exawind-time-$i.txt
   rm nalu-wind-time-$i.txt
   rm amr-wind-time-$i.txt
   rm tioga-time-$i.txt
    ((i=i+1))
done
