# Computing
import numpy as np
import torch
from model import PTModel
import time
from utils import predict, worldCoords, posFromDepth

# Image processing
from PIL import Image
from skimage.transform import resize

# Visualization
from matplotlib import pyplot as plt
import open3d as o3d
import visualizer


if __name__ == '__main__':
    # load model:
    pytorch_model = PTModel().float()
    checkpoint = torch.load("nyu.pth.tar", map_location=torch.device('cpu'))
    pytorch_model.load_state_dict(checkpoint['state_dict'])
    pytorch_model.eval()

    # Input images:
    file = "_in/sample_2.jpg"
    img = np.asarray(Image.open(file), dtype='float32')
    rgb_height, rgb_width = img.shape[:2]
    xx, yy = worldCoords(width=rgb_width // 2, height=rgb_height // 2)
    inputRGB = np.clip(img / 255, 0, 1)

    # Compute results
    start = time.time()
    output = predict(pytorch_model, inputRGB) # Here they did (1000 / predict()) / 1000 in demo file, absurd i know
    print(f"Predicted in {time.time() - start} s.")
   
    # save image:
    im = Image.fromarray(np.uint8(output * 255))
    im.save("_out/sample_2_depth.png")

    # Compute PCD and visualize:
    #output = resize(np.load('demo_depth.npy'), (240,320))# original demo
    pcd = posFromDepth(output.copy(), xx, yy)
    pcd = pcd.astype('float32')

    # SHOW intensity based pcl, not working well, need to do the dividing thing in the predict line, 
    # and still, the resulting pcl is not that correct
    '''
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(pcd)
    # Flip it, otherwise the pointcloud will be upside down
    pcl.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcl])
    '''

    # SHOW colored pcd with open3d, works fine if you resize the rgb to the same shape as depth frame
    # AND no need to divide in the 'predict' line above

    #inputRGB = Image.open('_in/sample_2.jpg').resize((320, 240)) # resizing in PIL
    #inputRGB.save('_in/sample3.jpg')
    inputRGB = o3d.io.read_image('_in/sample3.jpg')
    output = o3d.io.read_image('_out/sample_2_depth.png')

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        inputRGB, output, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # Visualize normal mode
    #o3d.visualization.draw_geometries([pcd])

    # Visualize with rotation and save frames to make a gif later
    visualizer.custom_draw_geometry_with_rotation(pcd)

    # display:
    # plt.imshow(output)
    # plt.show()
