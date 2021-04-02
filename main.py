import numpy as np
from matplotlib import pyplot as plt
import torch
from model import PTModel
from PIL import Image
import time
from skimage.transform import resize
from utils import predict, worldCoords, posFromDepth

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
    input = np.clip(img / 255, 0, 1)

    # Compute results
    start = time.time()
    output = predict(pytorch_model, input)
    print(f"Predicted in {time.time() - start} s.")

    # Compute PCD and visualize:
    # output = resize(np.load('demo_depth.npy'), (240,320))# original demo
    pcd = posFromDepth(output.copy(), xx, yy)
    pcd = pcd.astype('float32')

    import open3d as o3d
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(pcd)
    o3d.visualization.draw_geometries([pcl])

    # save image:
    # im = Image.fromarray(np.uint8(output * 255))
    # im.save("_out/sample_2_depth.png")

    # display:
    # plt.imshow(output)
    # plt.show()
