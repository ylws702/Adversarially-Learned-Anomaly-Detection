#!/bin/bash
conda activate tf16
SCRIPT_PATH=$(realpath $0)
ALAD_BASE=$(dirname $(dirname $SCRIPT_PATH))
export LD_LIBRARY_PATH=/home/Derick/miniconda3/envs/tf16/lib:$LD_LIBRARY_PATH
LOG_PATH="$ALAD_BASE/log/$(date +'%Y%m%d/%H%M%S').log"
mkdir -p $(dirname $LOG_PATH)
rm "${ALAD_BASE}/train_logs/sony" -r
python $ALAD_BASE/main.py anogan sony run 2>&1 | tee $LOG_PATH
