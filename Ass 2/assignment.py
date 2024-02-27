import glm
import numpy as np
from tqdm import tqdm
import Assignment_2 as a2
import settings as settings
import cv2 as cv
import pickle
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from engine.config import config

from skimage import measure

block_size = 1.0
resolution = settings.grid_tile_size

def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
            colors.append([1.0, 1.0, 1.0] if (x+z) % 2 == 0 else [0, 0, 0])
    return data, colors


def set_voxel_positions(width, height, depth):
    # Generates random voxel locations
    # TODO: You need to calculate proper voxel arrays instead of random ones.
    data, colors = [], []
    foreground_mask = []
    for n_camera in range(1, settings.num_cameras+1):
        foreground_mask.append(cv.imread(f'data/cam{n_camera}/foreground_mask.jpg', 0))
    voxel_volume = np.zeros((width, height, depth, settings.num_cameras), dtype=bool)
    lookup_table = create_lookup_table(voxel_volume)

    visible_per_camera = np.zeros(settings.num_cameras)
    # Iterate over pixels in the lookup table
    for voxel_coords, n_camera, pixel_coords in lookup_table:
        # Check if pixel is foreground in the corresponding view
        if is_foreground(pixel_coords, foreground_mask[n_camera-1]):
            # Mark the voxel visible for this view
            # print(f'Voxel at {voxel_coords} is visible in camera {n_camera}')
            # if n_camera == 1 or n_camera == 2 or n_camera == 3 or n_camera == 4:
            #     data.append([voxel_coords[0]*block_size - width/2, voxel_coords[2]*block_size, voxel_coords[1]*block_size - depth/2])
            #     colors.append([voxel_coords[0] / width, voxel_coords[2] / depth, voxel_coords[1] / height])
            visible_per_camera[n_camera-1] += 1
            voxel_volume[voxel_coords[0], voxel_coords[1], voxel_coords[2], (n_camera-1)] = True
    
    # TEST CODE TO COUNT VISIBLE VOXELS PER CAMERA
    for c in range(settings.num_cameras):
        print(f"Camera {c+1}: {visible_per_camera[c]} visible voxels")

    visible_all_cameras = 0
    # Iterate over voxels to mark them visible if visible in all views
    voxels = np.all(voxel_volume, axis=3)
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                if voxels[x, y, z]:
                    # voxel_volume[x, y, z] = True
                    # print(f'Voxel at {x, y, z} is visible in all views')
                    visible_all_cameras += 1
                    data.append([(x*block_size - width/2) - resolution/2, (z*block_size), (y*block_size - depth/2) - resolution/2])
                    colors.append([x / width, z / depth, y / height])
    
    
    print(f"Total voxels visible in all cameras: {visible_all_cameras}")

    # for x in range(width):
    #     for y in range(height):
    #         for z in range(depth):
    #             if random.randint(0, 1000) < 5:
    #                 data.append([x*block_size - width/2, y*block_size, z*block_size - depth/2])
    #                 colors.append([x / width, z / depth, y / height])
    print("Start Marching Cube")
    mesh(voxels)
    print("End Marching Cube")
    return data, colors


def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    cameraposition = np.zeros((4, 3, 1))
    for c in range(1, settings.num_cameras+1):
        _, _, rvecs, tvecs = a2.read_camera_parameters(c)
        print(rvecs, tvecs)
        rotM = cv.Rodrigues(rvecs)[0]
        cameraposition[(c-1)] = (-np.dot(np.transpose(rotM), tvecs / settings.grid_tile_size))

    cameraposition2 = [[cameraposition[0][0][0], -cameraposition[0][2][0], cameraposition[0][1][0]],
                       [cameraposition[1][0][0], -cameraposition[1][2][0], cameraposition[1][1][0]],
                       [cameraposition[2][0][0], -cameraposition[2][2][0], cameraposition[2][1][0]],
                       [cameraposition[3][0][0], -cameraposition[3][2][0], cameraposition[3][1][0]]]

    # Different colors are assigned to each of the cameras
    colors = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]
    return cameraposition2, colors


def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    cam_angles = [[0, 45, -45], [0, 135, -45], [0, 225, -45], [0, 315, -45]]
    cam_rotations = [glm.mat4(1), glm.mat4(1), glm.mat4(1), glm.mat4(1)]
    for c in range(len(cam_rotations)):
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])
    return cam_rotations


# Our extra functions

def create_lookup_table(voxel_volume):
    lookup_table = []

    # Retrieve the variable from the pickle file if lookup table is already created TODO: Remove!!!
    if os.path.exists('lookup_table.pkl'):
        with open('lookup_table.pkl', 'rb') as f:
            return pickle.load(f)
        
    (heightImg, widthImg) = cv.imread(f'data/cam1/foreground_mask.jpg', 0).shape # TODO: Generalise to all the cameras but without having it do it for each iteration?
    for x in tqdm(range(voxel_volume.shape[0]), desc="Lookup Table Progress"):
        for y in range(voxel_volume.shape[1]):
            for z in range(voxel_volume.shape[2]):
                voxel_point = np.array([[(x*block_size - voxel_volume.shape[0]/2) * resolution, (y*block_size - voxel_volume.shape[1]/2) * resolution, -z*block_size * resolution]], dtype=np.float32)

                for c in range(1, settings.num_cameras+1):
                    camera_matrix, distortion_coeffs, rotation_vector, translation_vector = a2.read_camera_parameters(c)
                    # Project voxel point onto image plane of camera c
                    img_point, _ = cv.projectPoints(voxel_point, rotation_vector, translation_vector, camera_matrix, distortion_coeffs)
                    img_point = np.reshape(img_point, 2)
                    img_point = img_point[::-1]
                    

                    # Only accept pixels inside the image
                    if 0 <= img_point[0] < heightImg and 0 <= img_point[1] < widthImg:
                        # Store {XV, YV, ZV}, c and {xim, yim} in the look-up table
                        lookup_table.append((((x, y, z), c, img_point)))


    # TEST CODE TO COUNT ENTRIES IN LOOKUP TABLE
    # Dictionary to store counts for each camera
    camera_counts = np.zeros(settings.num_cameras)
    
    # Iterate over the lookup table
    for entry in lookup_table:
        (_, c, _) = entry
        # Increment the count for the camera index
        camera_counts[c-1] += 1
    
    for c in range(settings.num_cameras):
        print(f"Camera {c+1}: {camera_counts[c]} entries")

    # Save the variable to a pickle file
    with open('lookup_table.pkl', 'wb') as f:
        pickle.dump(lookup_table, f, protocol=4)

    return lookup_table

def is_foreground(pixel_coords, foreground_mask):
    # Check if pixel is foreground in the corresponding view
    y, x = int(pixel_coords[0]), int(pixel_coords[1]) # OpenCV uses (y, x) indexing
    return foreground_mask[y, x] == 255

def mesh(voxels):
    # Use marching cubes to obtain the surface mesh of these ellipsoids
    verts, faces, normals, values = measure.marching_cubes(voxels, 0)

    # Display resulting triangular mesh using Matplotlib. This can also be done
    # with mayavi (see skimage.measure.marching_cubes docstring).
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    ax.set_xlim(config["world_width"]-50/2, config["world_width"]+50/2)
    ax.set_ylim(-config["world_height"]-50/2, config["world_height"]+50/2)
    ax.set_zlim(0, 50)

    plt.tight_layout()
    plt.show(block=False)
    plt.waitforbuttonpress()
    plt.close()