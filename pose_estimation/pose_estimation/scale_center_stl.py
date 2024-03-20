import open3d as o3d
import numpy as np

def center_and_scale_stl(input_file, output_file, scale_factor):
    # Leer el archivo STL
    mesh = o3d.io.read_triangle_mesh(input_file)

    # Centrar el modelo
    mesh = center_mesh(mesh)

    # Escalar el modelo
    mesh = scale_mesh(mesh, scale_factor)
    # Calcular las normales del modelo
    mesh.compute_vertex_normals()

    # Guardar el modelo modificado en un nuevo archivo STL
    o3d.io.write_triangle_mesh(output_file, mesh)

def center_mesh(mesh):
    # Calcular el centroide del modelo
    centroid = np.asarray(mesh.get_center())

    # Mover el modelo para que su centroide esté en el origen
    mesh.translate(-centroid)

    return mesh

def scale_mesh(mesh, scale_factor):
    # Calcular el centroide del modelo después de la traslación
    centroid = np.asarray(mesh.get_center())

    # Escalar el modelo con respecto al centroide
    mesh.scale(scale_factor, center=centroid)

    return mesh

if __name__ == "__main__":
    # Archivo de entrada y salida
    pkg_path = '/home/enrico/pose_estimation_ws/src/pose_estimation'
    mesh_path = pkg_path+"/pose_estimation/red_pepper.stl"
    input_file = mesh_path
    output_file = "scaled_centered.stl"
    
    # Factor de escala (ajústalo según sea necesario)
    scale_factor = 0.001

    # Procesar el archivo STL
    center_and_scale_stl(input_file, output_file, scale_factor)

    print("Proceso completado. Archivo guardado como:", output_file)
