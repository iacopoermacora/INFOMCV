import glm
import random
import numpy as np
import Assignment_2 as a2
import settings as settings
import cv2 as cv

block_size = 1.0


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
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                if random.randint(0, 1000) < 5:
                    data.append([x*block_size - width/2, y*block_size, z*block_size - depth/2])
                    colors.append([x / width, z / depth, y / height])
    return data, colors


def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    cameraposition = np.zeros((4, 3, 1))
    for c in range(1, settings.num_cameras+1):
        _, _, rvecs, tvecs = a2.read_camera_parameters(c)
        print(rvecs, tvecs)
        rotM = cv.Rodrigues(rvecs)[0]
        cameraposition[(c-1)] = (-np.matrix(rotM).T * np.matrix(tvecs)) #tvecs /settings.square_size ???

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
