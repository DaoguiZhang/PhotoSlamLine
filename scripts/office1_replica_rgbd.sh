#!/bin/bash

# 2 3 4
for i in 0 1 2 3 4
do

../bin/replica_rgbd \
    ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
    ../cfg/ORB_SLAM3/RGB-D/Replica/office1.yaml \
    ../cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml \
    /home/zdg/zdg/CPlusPlusProjects/Photo-SLAM/data/Replica/office1 \
    ../results/replica_rgbd_$i/office1 \
    no_viewer
done
