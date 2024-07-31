import open3d as o3d
import numpy as np
        
def compare_pointclouds(pc1, pc2):
	pcd_tree = o3d.geometry.KDTreeFlann(pc2)
	distances = []
	for point in pc1.points:
		[k, idx, _] = pcd_tree.search_knn_vector_3d(point, 1)
		if k == 1:
			dist = np.linalg.norm(np.asarray(pc2.points)[idx[0]] - point)
			distances.append(dist)            
	return np.array(distances)
        
pcd1 = o3d.io.read_point_cloud("./data/base.ply")
pcd2 = o3d.io.read_point_cloud("./data/2.ply")
        
# Rotate imported point cloud so that it aligns with base point cloud
# These values are pre-computed using ICP. Each case is different. 
pose_matrix = [[0.937361657619, 0.026231795549, 0.347368687391, 0.158131882548],
        [-0.027551226318, 0.999619722366, -0.001141030341, 0.067797377706],
        [-0.347266525030, -0.008500874974, 0.937727928162, 0.024652980268],
        [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]]
        
pcd2 = pcd2.transform(pose_matrix)
distances = compare_pointclouds(pcd1, pcd2)
rmse = np.sqrt(np.mean(distances**2))
print("RMSE between aligned point clouds: {rmse}")
