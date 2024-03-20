# Author: Enrico Mendez
# Date: 06 Noviembre 2023
# Description: node for testing
from typing import Iterable, List, NamedTuple, Optional
import open3d as o3d
from ament_index_python import get_package_share_directory
import os
from sensor_msgs.msg import PointCloud2, PointField
import std_msgs.msg
import struct
import numpy as np
import ros2_numpy as r2n
import copy
import rospkg
import time
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock

class Class_name(Node):
    def __init__(self):
        super().__init__('Node_name')
        self.get_logger().info('Node_name initialized')
        # self.on_shutdown(self.cleanup)

        # Create variables
        self.scene_pointcloud= PointCloud2()
        self.segmented_point_cloud = PointCloud2()
        self.calc_pose = PointCloud2()
        self.start = True
        
        # Define constants
        self.clock = Clock()
        pkg_path = self.get_pkg_path(target='pose_estimation')
        mesh_path = pkg_path+"/scaled_centered.stl"
        number_of_points = 10000 # Ajust this number as you find it convenient.
        ideal_pepper = o3d.io.read_triangle_mesh(mesh_path)
        self.ideal_pepper = ideal_pepper.sample_points_poisson_disk(number_of_points)
        self.timer_period = 0.5
        self.timer = self.create_timer(self.timer_period, self.processing_method)


        # Create publishers
        self.pub_rgbd = self.create_publisher(PointCloud2,'realsense_points',10)
        self.msg_rgbd = PointCloud2()
        self.pub_segmented_img = self.create_publisher(PointCloud2,'realsense_segmented',10)
        self.msg_segmented_img = PointCloud2()
        self.pub_pose_estimation = self.create_publisher(PointCloud2,'calculated_pose',10)
        self.msg_pose_estimation = PointCloud2()

        
        # Create subscribers
        self.sub_point_cloud = self.create_subscription(PointCloud2,'/camera/depth/color/points', self.point_cloud_callback, 10)

    def get_pkg_path(self,target='size_estimation'):
        # Get exc path
        pkg_path = get_package_share_directory(target)

        # Converting to list
        parts = pkg_path.split('/')

        # Directing to the src folder
        replace = 'install'
        idx = parts.index(replace)
        parts[idx] = 'src'
        parts.remove('share')

        # Converting back to string
        path = '/'.join(parts)

        return path

    def ros2o3d(self, points):
        # Extraer las coordenadas xyz
        xyz = points[:, :3]

        # Crear un objeto PointCloud en Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        return pcd

    def o3d2ros(self, pcd, frame_id='map'):
        # Crear un nuevo mensaje PointCloud2
        new_msg = PointCloud2()

        # Establecer el encabezado del mensaje
        new_msg.header.stamp = self.get_clock().now().to_msg()
        new_msg.header.frame_id = frame_id

        # Convertir la nube de puntos de Open3D a un mensaje PointCloud2 de ROS2
        new_msg.height = 1
        new_msg.width = len(pcd.points)
        new_msg.fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                          PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                          PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)]
        new_msg.is_bigendian = False
        new_msg.point_step = 12
        new_msg.row_step = new_msg.point_step * len(pcd.points)
        new_msg.is_dense = True
        new_msg.data = np.array(pcd.points).astype(np.float32).tobytes()

        return new_msg

    def centering(self,point_cloud):
        pepper_centroid = np.mean(np.asarray(point_cloud.points), axis=0)

        # Translate point cloud to make the centroid the origin
        translated_points = np.asarray(point_cloud.points) - pepper_centroid
        point_cloud.points = o3d.utility.Vector3dVector(translated_points)
        return point_cloud
    
    def scale(self,point_cloud,scale_factor = 1,reference=np.array([0, 0, 0])):
        scaled_model = point_cloud.scale(scale_factor, reference)
        return scaled_model

    def point_cloud_callback(self,msg):
        # Convertir el mensaje PointCloud2 a una matriz NumPy
        points = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, msg.point_step // 4)

        # Realizar la transformación de la nube de puntos
        pcd = self.ros2o3d(points)

        # Visualizar la nube de puntos transformada
        # o3d.visualization.draw_geometries([pcd])


        self.scene_pointcloud = self.centering(pcd)
        segmented_points = self.color_segmentation(self.scene_pointcloud)
        self.segmented_point_cloud = segmented_points
        self.start = False
    
    def transform_point_cloud(self,points):
        # Extraer las coordenadas xyz
        xyz = points[:, :3]

        # Crear un objeto PointCloud en Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        return pcd
    
    def rospc_to_o3dpc(self, msg):
        # Convertir el mensaje PointCloud2 a una matriz NumPy
        points = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, msg.point_step // 4)

        # Extraer las coordenadas xyz
        xyz = points[:, :3]

        # Crear un objeto PointCloud en Open3D
        o3d_pc = o3d.geometry.PointCloud()
        o3d_pc.points = o3d.utility.Vector3dVector(xyz)
        o3d.visualization.draw_geometries(o3d_pc,window_name='title')
        return o3d_pc
        
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
        # self.draw_registration_result(source, target, np.identity(4),title='Initial pose')

        source_down, source_fpfh = self.preprocess_point_cloud(source, voxel_size)
        target_down, target_fpfh = self.preprocess_point_cloud(target, voxel_size)
        return source, target, source_down, target_down, source_fpfh, target_fpfh

    def draw_registration_result(self,source, target, transformation,title='Open3D'):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp],window_name=title)
        # Note: Since the functions "transform" and "paint_uniform_color" change the point cloud,
        # we call copy.deep to make copies and protect the original point clouds.
        return

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
    
    def o3dpc_to_rospc(self,src, frame_id = 'map'):
        # Convertir la nube de puntos de Open3D a un mensaje PointCloud2 de ROS2
        points = np.asarray(src.points)
        msg = PointCloud2()
        msg.header.frame_id = frame_id  # Cambia 'map' por el marco de referencia deseado
        msg.height = 1
        msg.width = points.shape[0]
        msg.fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                      PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                      PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * points.shape[0]
        msg.is_dense = True
        msg.data = points.tobytes()
        return msg

    def processing_method(self):
        if self.start:
            print('Waiting ...')
            return
        self.get_logger().info("Transforming open3D pc to Point Cloud2 msg...")
        complete_scene = self.o3d2ros(self.scene_pointcloud,frame_id="camera_depth_frame")
        self.pub_rgbd.publish(complete_scene)
        # segmented_msg = self.o3dpc_to_rospc(self.segmented_point_cloud,frame_id="camera_depth_frame")
        # self.pub_segmented_img.publish(segmented_msg)
        # self.calc_pose = self.pose_register(self.ideal_pepper,self.segmented_point_cloud)
        # pose_msg = self.o3dpc_to_rospc(self.calc_pose,frame_id="camera_depth_frame")
        # self.pub_pose_estimation.publish(pose_msg)

def main(args=None):
    # Required lines for any node
    rclpy.init(args=args)
    node = Class_name()
    rclpy.spin(node)
    # Optional but good practices
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
