#!/bin/bash
mkdir -p checkpoints
python -u train.py --name raft-molecular --stage molecular --validation molecular --gpus 0 --num_steps 2000 --batch_size 10 --lr 0.0004 --image_size 256 256 --wdecay 0.0001