#!/usr/bin/env python3

# Author: Enrico Mendez
# Date: 06 Noviembre 2023
# Description: node for testing

import rospy
import open3d as o3d
import os
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2
import open3d_ros_helper as orh
import std_msgs.msg
import struct
import numpy as np
import copy
import rospkg
import time

class visual:
    def __init__(self):
        ### Node init ###
        time.sleep(5)
        rospy.init_node("visual_node", anonymous=False)
        rospy.loginfo("Starting visual.")
        ### Variables ###
        self.scene_pointcloud= PointCloud2()
        self.segmented_point_cloud = PointCloud2()
        self.calc_pose = PointCloud2()
        self.start = True

        ### Constants ###
        rospack = rospkg.RosPack()
        rospack.list() 
        pkg_path = str(rospack.get_path('pose_register'))
        mesh_path = pkg_path+"/scripts/red_pepper.stl"
        number_of_points = 10000 # Ajust this number as you find it convenient.
        ideal_pepper = o3d.io.read_triangle_mesh(mesh_path)
        self.ideal_pepper = ideal_pepper.sample_points_poisson_disk(number_of_points)
        
        ### Subscribers ###
        rospy.Subscriber("/camera/depth/color/points", PointCloud2, self.camera_processing)
                
        ### Publishers ###
        self.rgbd_pub = rospy.Publisher("realsense_points", PointCloud2, queue_size=10)
        self.segmented_img_pub = rospy.Publisher("realsense_segmented", PointCloud2, queue_size=10)
        self.pose_pc_pub = rospy.Publisher("Calculated_pose", PointCloud2, queue_size=10)
    
    def centering(self,point_cloud):
        pepper_centroid = np.mean(np.asarray(point_cloud.points), axis=0)

        # Translate point cloud to make the centroid the origin
        translated_points = np.asarray(point_cloud.points) - pepper_centroid
        point_cloud.points = o3d.utility.Vector3dVector(translated_points)
        return point_cloud
    
    def scale(self,point_cloud,scale_factor = 1,reference=np.array([0, 0, 0])):
        scaled_model = point_cloud.scale(scale_factor, reference)
        return scaled_model
    
    def pepper_pre_process(self):
        rospy.loginfo("Centering point cloud...")
        self.ideal_pepper = self.centering(self.ideal_pepper)
        rospy.loginfo("Scaling point cloud...")
        self.ideal_pepper = self.scale(self.ideal_pepper,scale_factor=0.001)

    def camera_processing(self,data):
        self.scene_pointcloud = self.centering(orh.rospc_to_o3dpc(data))
        segmented_points = self.color_segmentation(self.scene_pointcloud)
        # o3d.visualization.draw_geometries([segmented_points])
        self.segmented_point_cloud = segmented_points
        self.start = False

    def color_segmentation(self,point_cloud):
        blue_lower = [0, 0, 100]
        blue_upper = [40, 200, 255]
        colors = np.asarray(point_cloud.colors) * 255
        


        is_blue = np.logical_and.reduce(
            [colors[:, 0] >= blue_lower[0], colors[:, 0] <= blue_upper[0],
            colors[:, 1] >= blue_lower[1], colors[:, 1] <= blue_upper[1],
            colors[:, 2] >= blue_lower[2], colors[:, 2] <= blue_upper[2]]
        )
        blue_points = point_cloud.select_by_index(np.where(is_blue)[0])
        return blue_points

    ### Functions for pose register ###

    def draw_registration_result(self,source, target, transformation,title='Open3D'):
        return
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp],window_name=title)
        # Note: Since the functions "transform" and "paint_uniform_color" change the point cloud,
        # we call copy.deep to make copies and protect the original point clouds.

    def preprocess_point_cloud(self,pcd, voxel_size):
        print(":: Downsample with a voxel size of %.3f." % voxel_size)
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 2
        print(":: Estimate normal with  search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        )
        radius_feature = voxel_size * 5
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down, 
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )
        return pcd_down, pcd_fpfh
    
    def prepare_dataset(self,voxel_size,source,target):
        print(":: Load two point clouds and disturb the initial pose.")
        trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], 
                                [1.0, 0.0, 0.0, 0.0], 
                                [0.0, 1.0, 0.0, 0.0], 
                                [0.0, 0.0, 0.0, 1.0]])
        source.transform(trans_init)
        self.draw_registration_result(source, target, np.identity(4),title='Initial pose')

        source_down, source_fpfh = self.preprocess_point_cloud(source, voxel_size)
        target_down, target_fpfh = self.preprocess_point_cloud(target, voxel_size)
        return source, target, source_down, target_down, source_fpfh, target_fpfh
    
    def execute_global_registration(self,source_down, target_down, source_fpfh, target_fpfh, voxel_size):
        distance_threshold = voxel_size * 1.5
        print(":: RANSAC registration on downsampled point clouds. ")
        print("   Since the downsampling voxel sixe is %.3f, " % voxel_size)
        print("   we use a liberal distance threshold %.3f." % distance_threshold)
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True, 
            distance_threshold, 
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3, 
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9
                ),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold
                )
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.999)
        )
        return result
    
    def refine_registration(self,source, target, source_fpfh, target_fpfh, voxel_size,result_ransac):
        distance_threshold = voxel_size * 0.4
        print(":: Point-to-plane ICP registration is applied on original point")
        print("   clouds to refine the alignment. This time we use a strict")
        print("   distance threshold %.3f." % distance_threshold)
        result = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, result_ransac.transformation, 
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        return result
    
    def compute_normals(self,pcd, radius):
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
        )
    
    ### Pose register functions end ###

    def pose_register(self,source,target):
        voxel_size = 0.005 # means 5cm for the original dataset... gotta check on mine
        source, target, source_down, target_down, source_fpfh, target_fpfh = self.prepare_dataset(
            voxel_size,source,target)
        result_ransac = self.execute_global_registration(source_down, target_down, 
                                            source_fpfh, target_fpfh, 
                                            voxel_size)
        print(result_ransac)
        self.draw_registration_result(source_down, target_down, result_ransac.transformation,title='Ransac aprox')
        radius_normal = voxel_size * 2
        self.compute_normals(source, radius_normal)
        self.compute_normals(target, radius_normal)
        result_icp = self.refine_registration(source, target, source_fpfh, target_fpfh, voxel_size,result_ransac)
        print(result_icp)
        calc_pose = source.transform(result_icp.transformation)
        self.draw_registration_result(source, target, result_icp.transformation,title='ICP estimation')
        return calc_pose

    def main_loop(self):
        rospy.on_shutdown(self.cleanup)
        r = rospy.Rate(20)
        self.pepper_pre_process()
        rospy.loginfo("Transforming open3D pc to Point Cloud2 msg...")
        rospy.loginfo("Publishing msg.")
        
        while not rospy.is_shutdown():
            if self.start:
                print('Waiting')
                continue
            rospy.loginfo("Pose register init")
            print("begin")
            rospy.loginfo("Transforming open3D pc to Point Cloud2 msg...")
            complete_scene = orh.o3dpc_to_rospc(self.scene_pointcloud,frame_id="camera_depth_frame")
            self.rgbd_pub.publish(complete_scene)
            segmented_msg = orh.o3dpc_to_rospc(self.segmented_point_cloud,frame_id="camera_depth_frame")
            self.segmented_img_pub.publish(segmented_msg)
            self.calc_pose = self.pose_register(self.ideal_pepper,self.segmented_point_cloud)
            pose_msg = orh.o3dpc_to_rospc(self.calc_pose,frame_id="camera_depth_frame")
            self.pose_pc_pub.publish(pose_msg)
            # rospy.signal_shutdown('Done')
            r.sleep()

    def cleanup(self):
        # Before killing node
        rospy.loginfo("Shutting down visual.")

if __name__ == "__main__":
    visual_node = visual()
    visual_node.main_loop()