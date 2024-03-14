import rclpy
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
import open3d as o3d
import time

class PointCloudHandler:
    def __init__(self):
        self.node = rclpy.create_node('pointcloud_handler')
        self.subscriber = self.node.create_subscription(PointCloud2, '/camera/depth/color/points', self.callback, 10)
        self.publisher = self.node.create_publisher(PointCloud2, 'this_is_new', 10)

    def ros2o3d(self, points):
        # Extraer las coordenadas xyz
        xyz = points[:, :3]

        # Crear un objeto PointCloud en Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        return pcd

    def o3d2ros(self, pcd, frame_id='camera_depth_optical_frame'):
        # Crear un nuevo mensaje PointCloud2
        new_msg = PointCloud2()

        # Establecer el encabezado del mensaje
        new_msg.header.stamp = self.node.get_clock().now().to_msg()
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

    def callback(self, msg):
        # Convertir el mensaje PointCloud2 a una matriz NumPy
        points = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, msg.point_step // 4)

        # Realizar la transformación de la nube de puntos
        pcd = self.ros2o3d(points)

        # Visualizar la nube de puntos transformada
        o3d.visualization.draw_geometries([pcd])

        # Convertir la nube de puntos de Open3D a un mensaje PointCloud2 de ROS2
        new_msg = self.o3d2ros(pcd)

        # Publicar la nueva nube de puntos
        self.publisher.publish(new_msg)
        self.node.get_logger().info('Publicando nube de puntos transformada')

        # Añadir un pequeño tiempo de espera para asegurarse de que el mensaje se publique correctamente
        time.sleep(0.1)

def main(args=None):
    rclpy.init(args=args)
    pointcloud_handler = PointCloudHandler()
    executor = MultiThreadedExecutor()
    executor.add_node(pointcloud_handler.node)
    executor.spin()
    executor.shutdown()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
