#!/bin/bash
cd NTU_SNTL_Task3
rm -rf __pycache__
python main.py --exp DCASE2020_baseline --project DCASE2020 --nb_epochs 5 --dnorm



