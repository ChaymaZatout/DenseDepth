'''
 This script as well as the visualizer and some additions to the Utils script are the only differences
 between the orignal repo and this one.
'''

# Computing
import numpy as np
import torch
from model import PTModel
import time
from utils import predict, worldCoords, posFromDepth, scale_up

# Image processing
from PIL import Image
from skimage.transform import resize
from skimage.filters import median
from skimage.morphology import disk

# Visualization
from matplotlib import pyplot as plt
import open3d as o3d
import visualizer


if __name__ == '__main__':
    # load model:
    pytorch_model = PTModel().float()
    checkpoint = torch.load("data/nyu.pth.tar", map_location=torch.device('cpu'))
    pytorch_model.load_state_dict(checkpoint['state_dict'])
    pytorch_model.eval()

    # Load input imgs, not adapted for a video yet:
    file = "data/_in/classroom__rgb_00283.jpg"
    img = np.asarray(Image.open(file), dtype='float32')
    rgb_height, rgb_width = img.shape[:2]
    xx, yy = worldCoords(width=rgb_width, height=rgb_height)
    inputRGB = np.clip(img / 255, 0, 1)

    # Compute results
    start = time.time()
    output = predict(pytorch_model, inputRGB)
    print(f"Predicted in {time.time() - start} s.")
   
    # save image:
    # im = Image.fromarray(np.uint8(output * 255))
    # im.save("data/_out/sample_2_depth.png")

    # Compute PCD and visualize:
    output = scale_up(2, output) * 10.0
    pcd = posFromDepth(output.copy(), xx, yy)
    
    # Open3d pcd:
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(pcd)
    pcl.colors = o3d.utility.Vector3dVector(inputRGB.reshape(rgb_width * rgb_height, 3))
    # Flip it, otherwise the pointcloud will be upside down
    pcl.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # Compute point cloud from rgbd image:
    '''
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
         inputRGB, output, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
         rgbd_image,
         o3d.camera.PinholeCameraIntrinsic(
             o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    '''
    # Visualize normal mode
    o3d.visualization.draw_geometries([pcl])
    # Visualize with rotation and save frames to make a gif later
    #visualizer.custom_draw_geometry_with_rotation(pcl)

    # display:
    # plt.imshow(output)
    # plt.show()

