python ./partkey.py --dataset_center_path /home/zdg/zdg/CPlusPlusProjects/Photo-SLAM/data_new  --result_main_folder /home/zdg/zdg/CPlusPlusProjects/Photo-SLAM/results


python ./partkey_tum_rgbd.py --dataset_center_path /home/zdg/zdg/CPlusPlusProjects/Photo-SLAM/data_new  --result_main_folder /home/zdg/zdg/CPlusPlusProjects/Photo-SLAM/results
#这是渲染rgb图和深度图，两种类型的图，最后保存。


#获取渲染得到的高斯球，用于论文的展示：（路径加采样）
python viz_for_fusion.py -d /home/zdg/zdg/CPlusPlusProjects/Photo-SLAM/result_replica/PhotoSlamLine/test_structural_gaussian_sphere/replica_rgbd_0/office1/16588_shutdown/ply -s 1
