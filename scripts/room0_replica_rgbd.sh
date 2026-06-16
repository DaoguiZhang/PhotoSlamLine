#!/bin/bash

# 2 3 4
for i in 5
do
../bin/replica_rgbd \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/RGB-D/Replica/room0.yaml \
    ../cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml \
    /workspace/code/SEGS-SLAM/datasets/replica/room0 \
    ../results/replica_rgbd_$i/room0 \
    no_viewer
done
