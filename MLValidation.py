import os
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import numpy as np
import open3d as o3d
from data_utils.indoor3d_util import g_label2color
    
    
# Define the classes and mappings
classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {i: cat for i, cat in enumerate(seg_classes.keys())}
    
def add_vote(vote_label_pool, point_idx, pred_label, weight):
	B = pred_label.shape[0]
	N = pred_label.shape[1]
	for b in range(B):
		for n in range(N):
			if weight[b, n] != 0 and not np.isinf(weight[b, n]):
				vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
	return vote_label_pool
    
def load_point_cloud(file_path):
	pcd = o3d.io.read_point_cloud(file_path)
	points = np.asarray(pcd.points)
	labels = np.zeros((points.shape[0],))  # Assuming no labels in the .ply file
	return points, labels
    
def setup_logging(log_dir):
	logger = logging.getLogger("Model")
	logger.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	file_handler = logging.FileHandler(log_dir)
	file_handler.setLevel(logging.INFO)
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)
	return logger
    
def save_colored_obj(filename, points, labels, colors):
	with open(filename, 'w') as four:
		for i in range(points.shape[0]):
			color = colors[labels[I]]
			fout.write('v %f %f %f %d %d %d\n' % (points[i, 0], points[i, 1], points[i, 2], color[0], color[1], color[2]))
    
def generate_colored_png(filename, points, labels, colors):
	img_size = 500
	img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    
	x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
	y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
	points[:, 0] = (points[:, 0] - x_min) / (x_max - x_min) * (img_size - 1)
	points[:, 1] = (points[:, 1] - y_min) / (y_max - y_min) * (img_size - 1)
    
	for i in range(points.shape[0]):
		x = int(points[i, 0])
		y = int(points[i, 1])
		img[y, x, :] = colors[labels[I]]
    
	o3d.io.write_image(filename, o3d.geometry.Image(img))
    
# Define hyperparameters and setup
batch_size = 64
num_point = 2048
    
log_dir = 'log/sem_seg/pointnet2_sem_seg/eval.txt'
visual_dir = Path('log/sem_seg/pointnet2_sem_seg/visual/')
visual_dir.mkdir(parents=True, exist_ok=True)
    
#logger = setup_logging(log_dir)
    
NUM_CLASSES = 13
    
# Load the single scene
test_scene = '5-000.ply'
whole_scene_data, whole_scene_label = load_point_cloud(os.path.join(test_scene))
print(f"The number of points in the test scene: {len(whole_scene_data)}")
    
# Load model
experiment_dir = 'log/sem_seg/pointnet2_sem_seg'
model_name = 'pointnet2_sem_seg'
MODEL = importlib.import_module(model_name)
classifier = MODEL.get_model(NUM_CLASSES).cuda()
checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
classifier.load_state_dict(checkpoint['model_state_dict'])
classifier = classifier.eval()
    
# Evaluation
vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))
with torch.no_grad():
	for _ in tqdm(range(1), total=1):  # Single vote for simplicity
		num_blocks = (len(whole_scene_data) + num_point - 1) // num_point
		for sbatch in range(num_blocks):
			start_idx = sbatch * num_point
			end_idx = min((sbatch + 1) * num_point, len(whole_scene_data))
			real_batch_size = end_idx - start_idx
			
			batch_data = np.zeros((batch_size, num_point, 9))
			batch_label = np.zeros((batch_size, num_point))
			batch_point_index = np.zeros((batch_size, num_point))
			batch_smpw = np.ones((batch_size, num_point))
			
			# Ensure whole_scene_data has the correct shape
			padded_whole_scene_data = np.zeros((num_point, 9))
			padded_whole_scene_data[:real_batch_size, :3] = whole_scene_data[start_idx:end_idx, ...]
			
			batch_data[0:real_batch_size, ...] = padded_whole_scene_data
			
			# Ensure whole_scene_label has the correct shape
			padded_whole_scene_label = np.zeros(num_point)
			padded_whole_scene_label[:real_batch_size] = whole_scene_label[start_idx:end_idx]
			
			batch_label[0:real_batch_size, ...] = padded_whole_scene_label
			
			# Ensure batch_point_index has the correct shape
			padded_point_index = np.zeros(num_point)
			padded_point_index[:real_batch_size] = np.arange(start_idx, end_idx)
			
			batch_point_index[0:real_batch_size, ...] = padded_point_index
			
			batch_data[:, :, 3:6] /= 1.0

			torch_data = torch.Tensor(batch_data)
			torch_data = torch_data.float().cuda()
			torch_data = torch_data.transpose(2, 1)
			seg_pred, _ = classifier(torch_data)
			batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()
			
			vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
									   batch_pred_label[0:real_batch_size, ...],
									   batch_smpw[0:real_batch_size, ...])
    
pred_label = np.argmax(vote_label_pool, 1)
    
# Print label class and the number of points
label_counts = np.bincount(pred_label, minlength=NUM_CLASSES)
for label_index, count in enumerate(label_counts):
    print(f"Class:{classes[label_index]},:{count}")
    
# Save results
save_colored_obj('pleasework.obj', whole_scene_data, pred_label, g_label2color)
generate_colored_png('check.png', whole_scene_data, pred_label, g_label2color)
