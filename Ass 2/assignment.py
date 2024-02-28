import glm
import numpy as np

from tqdm import tqdm
import camera_calibration as cc
import settings as settings
import cv2 as cv
import pickle
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from engine.config import config

from skimage import measure

block_size = 1.0
resolution = settings.GRID_TILE_SIZE

def ray_trace(camera_position, voxel_positions, voxel_grid):
    """
    Determines which voxels are visible from the camera position.

    :param camera_position: The (x, y, z) coordinates of the camera.
    :param voxel_positions: A list of (x, y, z) coordinates for each voxel.
    :param voxel_grid: A 3D numpy array representing the presence (True) or absence (False) of voxels.
    :return: A list of booleans indicating visibility for each voxel in voxel_positions.
    """
    print("ray_trace")
    print("camera_position: ", camera_position)
    print("voxel grid shape: ", np.shape(voxel_grid))
    for target_voxel in voxel_positions:
        if voxel_grid[target_voxel[0], target_voxel[1], target_voxel[2]]:
            direction = target_voxel - camera_position
            distance_to_target = np.linalg.norm(direction)
            direction /= distance_to_target  # Normalize direction vector

            step_size = 1.0  # Adjust based on the desired precision
            num_steps = int(distance_to_target / step_size)

            is_first = True
            outside = True
            # for step in range(1, num_steps):
            step = 1
            while True:
                point_along_ray = camera_position + direction * step * step_size
                step += 1
                x, y, z = np.round(point_along_ray).astype(int)
                # print("Point along ray: ", point_along_ray)
                
                # Check if we're outside the bounds of the voxel grid
                if x < 0 or x >= voxel_grid.shape[0] or y < 0 or y >= voxel_grid.shape[1] or z < 0 or z >= voxel_grid.shape[2]:
                    if outside == False:
                        break
                    continue

                outside = False
                
                if voxel_grid[x, y, z]:  # There's a voxel blocking the view
                    if is_first:
                        is_first = False
                    else:
                        voxel_grid[x, y, z] = False

    return voxel_grid

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
    color_images = []
    for n_camera in range(1, settings.NUM_CAMERAS+1):
        video_path = f'data/cam{n_camera}/foreground_mask.avi'
        cap = cv.VideoCapture(video_path)
        frame_number = np.min([int(cap.get(cv.CAP_PROP_FRAME_COUNT)), (settings.FRAME_NUMBER - 1)])
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)
        _, frame = cap.read()
        frame_cvt = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        foreground_mask.append(frame_cvt) # cv.imread(f'data/cam{n_camera}/foreground_mask.jpg', 0)

        video_path_color = f'data/cam{n_camera}/video.avi'
        cap_color = cv.VideoCapture(video_path_color)
        cap_color.set(cv.CAP_PROP_POS_FRAMES, frame_number)
        _, frame_color = cap_color.read()
        color_images.append(frame_color) # TODO: NEW
        cap.release()
        cap_color.release()
    voxel_volume = np.zeros((width, height, depth, settings.NUM_CAMERAS), dtype=bool)
    lookup_table = create_lookup_table(voxel_volume)
    voxel_to_colors = np.zeros((width, height, depth, settings.NUM_CAMERAS, 3), dtype=int) # TODO: NEW

    visible_per_camera = np.zeros(settings.NUM_CAMERAS)
    # Iterate over pixels in the lookup table
    for voxel_coords, n_camera, pixel_coords in lookup_table:
        # Check if pixel is foreground in the corresponding view
        if is_foreground(pixel_coords, foreground_mask[n_camera-1]):
            visible_per_camera[n_camera-1] += 1
            voxel_volume[voxel_coords[0], voxel_coords[1], voxel_coords[2], (n_camera-1)] = True
            color = color_images[n_camera-1][int(pixel_coords[1]), int(pixel_coords[0]), :]
            voxel_to_colors[voxel_coords[0], voxel_coords[1], voxel_coords[2], (n_camera-1)] = color # TODO: end NEW
    
    # TEST CODE TO COUNT VISIBLE VOXELS PER CAMERA
    for c in range(settings.NUM_CAMERAS):
        print(f"Camera {c+1}: {visible_per_camera[c]} visible voxels")

    visible_all_cameras = 0
    # Iterate over voxels to mark them visible if visible in all views
    voxels = np.all(voxel_volume, axis=3)
    voxel_color_visibility = np.zeros((settings.NUM_CAMERAS, width, height, depth), dtype=bool)
    voxels_for_colors = voxels.copy()
    for n_camera in range(1, settings.NUM_CAMERAS+1):
        # SET THE CORRECT PARAMETERS FOR EACH CAMERA HERE
        camera_position, _ = get_cam_positions()
        camera_position[n_camera-1][1], camera_position[n_camera-1][2] = camera_position[n_camera-1][2], camera_position[n_camera-1][1]
        # print("number of true values in voxels (before argwhere): ", np.count_nonzero(voxels))
        voxel_positions = np.argwhere(voxels)
        # print("number of true values in voxels (after argwhere): ", np.count_nonzero(voxels))
        voxel_color_visibility[n_camera-1] = ray_trace(camera_position[n_camera-1], voxel_positions, voxels_for_colors)
        # print("number of true values in voxel_color_visibility cam 1: ", np.count_nonzero(voxel_color_visibility[n_camera-1]))
        # print("number of true values in voxels: ", np.count_nonzero(voxels))
        # TODO: Define the output
    count_colors = 0
    count_black = 0
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                if voxels[x, y, z]:
                    # voxel_volume[x, y, z] = True
                    # print(f'Voxel at {x, y, z} is visible in all views')
                    visible_all_cameras += 1
                    data.append([(x*block_size - width/2) - resolution/2, (z*block_size), (y*block_size - depth/2) - resolution/2])
                    # colors.append([x / width, z / depth, y / height])
                    color_voxel_cam = []
                    for n_camera in range(1, settings.NUM_CAMERAS+1):
                        if voxel_color_visibility[n_camera-1, x, y, z]:
                            color_voxel_cam.append(voxel_to_colors[x, y, z, n_camera-1] / 255)
                    if len(color_voxel_cam) > 0:
                        print("color_voxel_cam size: ", len(color_voxel_cam))
                        # if len(color_voxel_cam) > settings.NUM_CAMERAS:
                            # print(f"Voxel at {x, y, z} has more than 4 colors")
                        # print("color_voxel_cam: ", color_voxel_cam)
                        # print("shape color_voxel_cam: ", np.shape(color_voxel_cam))
                        color_voxel = np.mean(color_voxel_cam, axis=0)
                        count_colors += 1
                    else:
                        color_voxel = [0, 0, 0]
                        count_black += 1
                    # if voxel_color_visibility[1][x, y, z]:
                    #     # Red voxel
                    #     color_voxel = [1, 0, 0]
                    # else:
                    #     # Black voxel
                    #     color_voxel = [0, 0, 0]
                    colors.append(color_voxel)
    
    
    print(f"Total voxels visible in all cameras: {visible_all_cameras}")
    print(f"Total colored voxels: {count_colors}")
    print(f"Total black voxels: {count_black}")

    # for x in range(width):
    #     for y in range(height):
    #         for z in range(depth):
    #             if random.randint(0, 1000) < 5:
    #                 data.append([x*block_size - width/2, y*block_size, z*block_size - depth/2])
    #                 colors.append([x / width, z / depth, y / height])
    mesh(voxels)
    print("Saved Mesh to voxel_mesh.png")
    return data, colors


def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    cameraposition = np.zeros((4, 3, 1))
    for c in range(1, settings.NUM_CAMERAS+1):
        _, _, rvecs, tvecs = read_camera_parameters(c)
        rotM = cv.Rodrigues(rvecs)[0]
        cameraposition[(c-1)] = (-np.dot(np.transpose(rotM), tvecs / settings.GRID_TILE_SIZE))

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
    
    (heightImg, widthImg) = cv.imread(f'data/cam1/background_model.jpg', 0).shape # TODO: Generalise to all the cameras but without having it do it for each iteration?
    for x in tqdm(range(voxel_volume.shape[0]), desc="Lookup Table Progress"):
        for y in range(voxel_volume.shape[1]):
            for z in range(voxel_volume.shape[2]):
                voxel_point = np.array([[(x*block_size - voxel_volume.shape[0]/2) * resolution, (y*block_size - voxel_volume.shape[1]/2) * resolution, -z*block_size * resolution]], dtype=np.float32)

                for c in range(1, settings.NUM_CAMERAS+1):
                    camera_matrix, distortion_coeffs, rotation_vector, translation_vector = read_camera_parameters(c)
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
    camera_counts = np.zeros(settings.NUM_CAMERAS)
    
    # Iterate over the lookup table
    for entry in lookup_table:
        (_, c, _) = entry
        # Increment the count for the camera index
        camera_counts[c-1] += 1
    
    for c in range(settings.NUM_CAMERAS):
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
    verts, faces, _, _ = measure.marching_cubes(voxels, 0)

    # Display resulting triangular mesh using Matplotlib. This can also be done
    # with mayavi (see skimage.measure.marching_cubes docstring).
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    ax.set_xlim((-config['world_width']+100)/2, (config['world_width']+100)/2)
    ax.set_ylim((-config['world_height']+100)/2, (config['world_height']+100)/2)
    ax.set_zlim(0, config['world_depth'])

    plt.tight_layout()
    plt.savefig('voxel_mesh.png')

def create_background_model_gmm(video_path):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening background video file")
        return None

    # Create the background subtractor object
    backSub = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply the background subtractor to each frame
        fgMask = backSub.apply(frame)
    
    # Retrieve the final background model after processing all frames
    background_model = backSub.getBackgroundImage()
    
    cap.release()
    # Save the background model
    folder_path = os.path.dirname(video_path)
    cv.imwrite(f'{folder_path}/background_model.jpg', background_model)
    return background_model

def background_subtraction(frame, background_model_path, optimal_thresholds):
    h_thresh, s_thresh, v_thresh = optimal_thresholds
    # Load the background model
    background_model = cv.imread(background_model_path)
    background_model_hsv = cv.cvtColor(background_model, cv.COLOR_BGR2HSV)

    # Convert frame to HSV
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Calculate the absolute difference
    diff = cv.absdiff(frame_hsv, background_model_hsv)

    # Threshold for each channel
    _, thresh_h = cv.threshold(diff[:,:,0], h_thresh, 255, cv.THRESH_BINARY) 
    _, thresh_s = cv.threshold(diff[:,:,1], s_thresh, 255, cv.THRESH_BINARY) 
    _, thresh_v = cv.threshold(diff[:,:,2], v_thresh, 255, cv.THRESH_BINARY)


    # Combine the thresholds 
    threshold_mask = cv.bitwise_and(thresh_v, cv.bitwise_and(thresh_h, thresh_s))


    # Dilation 1 to fill in gaps
    kernel_1 = np.ones((5,5), np.uint8)
    dilation_mask = cv.dilate(threshold_mask, kernel_1, iterations=1)

    # BLOB 1 DETECTION AND REMOVAL
    # Find contours
    contours_1, _ = cv.findContours(dilation_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area to identify blobs
    min_blob_area_1 = 1500  # Adjust this threshold as needed
    blobs_1 = [cnt for cnt in contours_1 if cv.contourArea(cnt) > min_blob_area_1]

    # Draw the detected blobs
    blob_mask = np.zeros_like(dilation_mask)
    cv.drawContours(blob_mask, blobs_1, -1, (255), thickness=cv.FILLED)

    dilation_mask_bitw = cv.bitwise_and(threshold_mask, blob_mask)

    return dilation_mask_bitw

def manual_segmentation_comparison(first_video_frame, background_model_path, manual_mask_path, steps=[50, 10, 5, 1]):
    optimal_thresholds = None
    optimal_score = float('inf')
    previous_step = 255
    optimal_thresholds = (0, 0, 0)

    for current_step in steps:  # Ensure a final fine-grained search
        print(f"\nSearching with step size {current_step}")

        # Adjust the search ranges based on the previous optimal thresholds
        search_ranges = {
            'hue': range(max(0, optimal_thresholds[0] - previous_step), min(256, optimal_thresholds[0] + previous_step), current_step),
            'saturation': range(max(0, optimal_thresholds[1] - previous_step), min(256, optimal_thresholds[1] + previous_step), current_step),
            'value': range(max(0, optimal_thresholds[2] - previous_step), min(256, optimal_thresholds[2] + previous_step), current_step),
        }

        # Nested loops for searching threshold values, wrapped with tqdm for progress tracking
        for h_thresh in tqdm(search_ranges['hue'], desc="Hue Progress"):
            for s_thresh in tqdm(search_ranges['saturation'], desc="Saturation Progress", leave=False):
                for v_thresh in tqdm(search_ranges['value'], desc="Value Progress", leave=False):
                    test_threshold = (h_thresh, s_thresh, v_thresh)
                    segmented = background_subtraction(first_video_frame, background_model_path, test_threshold) # TODO: Change to frame
                    xor_result = cv.bitwise_xor(segmented, cv.imread(manual_mask_path, 0))
                    score = cv.countNonZero(xor_result)

                    if score < optimal_score:
                        optimal_score = score
                        optimal_thresholds = (h_thresh, s_thresh, v_thresh)

        # Print the optimal thresholds found in this iteration
        print(f"Optimal thresholds after refinement with step {current_step}: Hue={optimal_thresholds[0]}, Saturation={optimal_thresholds[1]}, Value={optimal_thresholds[2]}")

        # Decrease step size for the next iteration
        previous_step = current_step

    print(f'Optimal thresholds: Hue={optimal_thresholds[0]}, Saturation={optimal_thresholds[1]}, Value={optimal_thresholds[2]}')

    return optimal_thresholds

def create_segmented_video(video_path, background_model_path, optimal_thresholds):
    
    folder_path = os.path.dirname(video_path)
    output_video_path = f'{folder_path}/foreground_mask.avi'  # Update with the desired output video file path
    if os.path.exists(output_video_path):
        print("here")
        return
    
    # Open the video file
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Get video properties
    fps = cap.get(cv.CAP_PROP_FPS)
    size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH) + 0.5),
        int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) + 0.5))

    # Create a VideoWriter object to write the modified frames to a new video
    fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
    # print("Fourcc: ", fourcc)
    out = cv.VideoWriter(output_video_path, fourcc, fps, size, isColor=False)

    while True:
        
        ret, frame = cap.read()
        if not ret:
            break

        segmented = background_subtraction(frame, background_model_path, optimal_thresholds)

        # Dilation to fill in gaps
        kernel_2 = np.ones((2, 2), np.uint8)
        dilation_mask_2 = cv.dilate(segmented, kernel_2, iterations=1)

        # BLOB DETECTION AND REMOVAL
        # Find contours
        contours_2, _ = cv.findContours(dilation_mask_2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area to identify blobs
        min_blob_area_2 = 20  # Adjust this threshold as needed
        blobs_2 = [cnt for cnt in contours_2 if cv.contourArea(cnt) > min_blob_area_2]

        # Draw the detected blobs
        blob_mask_2 = np.zeros_like(dilation_mask_2)
        cv.drawContours(blob_mask_2, blobs_2, -1, (255), thickness=cv.FILLED)

        segmented = blob_mask_2 # TODO: Continue the code here to save all the frames in a video file

        out.write(segmented)

        # cv.imshow('Foreground Mask', segmented)
        # cv.waitKey(0) 
        # cv.destroyAllWindows()
        # cv.waitKey(1)

        # cv.imwrite(f'data/cam{camera_number}/foreground_mask.jpg', segmented)
    
    cap.release()
    out.release()
    cv.destroyAllWindows()

def read_camera_parameters(camera_number):
    directory = f'data/cam{camera_number}'
    file_name = f"{directory}/config.xml"

    # Open a FileStorage object to read data from the XML file
    fs = cv.FileStorage(file_name, cv.FILE_STORAGE_READ)

    # Read parameters from the file
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("distortion_coefficients").mat()
    rvecs = fs.getNode("rotation_vectors").mat()
    tvecs = fs.getNode("translation_vectors").mat()

    fs.release()

    return camera_matrix, dist_coeffs, rvecs, tvecs

'''# Call the function to get the camera intrinsics and extrinsics for each camera
for camera_number in range(1, settings.NUM_CAMERAS+1):
    cc.get_camera_intrinsics_and_extrinsics(camera_number)
    # background model
    background_video_path = f'data/cam{camera_number}/background.avi'
    background_model = create_background_model_gmm(background_video_path)
    # manual subtraction
    manual_mask_path = f'data/cam{camera_number}/manual_mask.jpg'
    video_path = f'data/cam{camera_number}/video.avi'
    _, first_video_frame = cv.VideoCapture(video_path).read()
    background_model_path = f'data/cam{camera_number}/background_model.jpg'
    optimal_thresholds = manual_segmentation_comparison(first_video_frame, background_model_path, manual_mask_path, steps=[50, 10, 5, 1])
    create_segmented_video(video_path, background_model_path, optimal_thresholds)'''
    