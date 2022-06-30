#!/bin/bash
mkdir log
for (( i = 1; i<=10; i=i+1 )); do
python3.5 train-val.py --feapath=./data/VideoEmotion/tsnv3_rgb_pool/,./data/VideoEmotion/tsnv3_flow_pool,./data/VideoEmotion/audioset/ --dataset=VideoEmotion --numclasses=8 --split=${i} --gpu=1 --dtype=valence --seqlen=18 --numhidden=1024 --numlayers=1 --batchsize=32 --numepochs=50 --learate=0.00005 2>&1 |tee log/VE8_split=${i}.log
done
python3 calcResult.py --numclasses=8