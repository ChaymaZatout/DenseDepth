
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import os


mainPath = '_in/'

def custom_draw_geometry_with_rotation(pcd):

    custom_draw_geometry_with_rotation.index = -1
    custom_draw_geometry_with_rotation.vis = o3d.visualization.Visualizer()

    if not os.path.exists(mainPath + 'rgbFrames/'):
        os.makedirs(mainPath + 'rgbFrames/')
    if not os.path.exists(mainPath + 'depthFrames/'):
        os.makedirs(mainPath + 'depthFrames/')

    def rotate(vis):
        
        ctr = vis.get_view_control()
        glb = custom_draw_geometry_with_rotation

        if glb.index >= 0:
            print("Capture image {:05d}".format(glb.index))
            depth = vis.capture_depth_float_buffer(False)
            image = vis.capture_screen_float_buffer(False)
            plt.imsave(mainPath + 'depthFrames/{:05d}.png'.format(glb.index), np.asarray(depth), dpi=1)
            plt.imsave(mainPath + 'rgbFrames/{:05d}.png'.format(glb.index), np.asarray(image), dpi=1)

        glb.index = glb.index + 1
        ctr.rotate(5.0, 0.0)
        
        return False

    vis = custom_draw_geometry_with_rotation.vis
    vis.create_window()
    vis.add_geometry(pcd)
    vis.register_animation_callback(rotate)
    vis.run()
    vis.destroy_window()
