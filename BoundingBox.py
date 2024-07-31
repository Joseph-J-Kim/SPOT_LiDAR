import open3d as o3d
import numpy as np

# Load the PLY file
ply_file = "base.ply"
point_cloud = o3d.io.read_point_cloud(ply_file).paint_uniform_color([0, 0, 0])
        
# Perform radius outlier removal
cl, ind = point_cloud.remove_radius_outlier(nb_points=30, radius=0.1)
filtered_cloud = point_cloud.select_by_index(ind)
        
# Perform statistical outlier removal
cl, ind = filtered_cloud.remove_statistical_outlier(nb_neighbors=300, std_ratio=0.2)
filtered_cloud_2 = filtered_cloud.select_by_index(ind)
        
points = np.asarray(filtered_cloud_2.points)
        
# Create an oriented bounding box
oriented_bounding_box = filtered_cloud_2.get_oriented_bounding_box()
oriented_bounding_box.color = [0, 0, 1]
        
# Visualize the filtered point cloud and the bounding box
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(filtered_cloud_2)
vis.add_geometry(oriented_bounding_box)
        
render_option = vis.get_render_option()
render_option.point_size = 1.0
vis.run()
vis.destroy_window()
        
# Get the dimensions
extents = oriented_bounding_box.extent
width, height, length = extents
        
volume = width * length * height
        
print(f"width: {width}")
print(f"length: {length}")
print(f"height: {height}")
print(f"The volume of the Oriented Bounding Box is: {volume}")
