#!/bin/bash
LOG=/mnt/data/datasets/download_datacomp.log
nohup bash /mnt/data/code/KUEA/scripts/download_datacomp_medium.sh > $LOG 2>&1 &
echo "PID: $!"
echo "Log: $LOG"
echo "Monitor: tail -f $LOG"
echo "Check: ps aux | grep download_datacomp"
