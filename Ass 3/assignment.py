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

import color_clustering as col_cl
import math

block_size = settings.BLOCK_SIZE
frame_cnt = -1
if settings.VISUALIZE_REAL_COLORS == True:
    visualize_real = True
else:
    visualize_real = False
final_centers = []

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
    '''
    Generates the voxel positions for the 3D projection

    :param width: The width of the voxel grid
    :param height: The height of the voxel grid
    :param depth: The depth of the voxel grid

    :return: data, colors
    '''
    i = 0
    global total_voxels_color
    global total_visible_voxels_per_cam
    global total_visible_voxels_colors_per_cam
    global total_voxel_volume_cleaned
    global total_centers
    global total_labels
    global total_voxels

    global cam_color_models_offline

    data, colors = [], []
    # Create a lookup table to store the voxel coordinates and the corresponding pixel coordinates for each camera

    # Sets the index of the frame of the final video
    idx = frame_cnt

    # The voxels to project are the ones from the voxel volume
    data = total_voxels[idx]

    # Determine the labels of the clustered voxels
    final_labels = col_cl.online_phase(total_labels, total_voxels, total_visible_voxels_colors_per_cam, total_visible_voxels_per_cam, cam_color_models_offline, idx)

    # Visualise the real colors or the clustered colors based on the settings
    if visualize_real == True:
        data = []
        # Create a counter to store the number of visible voxels in all cameras
        visible_all_cameras = 0
        for x in tqdm(range(settings.WIDTH), desc="Voxel Projection - Visible Voxels"):
                for y in range(settings.HEIGHT):
                    for z in range(settings.DEPTH):
                        if total_voxel_volume_cleaned[idx][x, z, y]:
                            visible_all_cameras += 1
                            # Store the voxel coordinates to output
                            voxel_to_display = [x*block_size - settings.WIDTH/2, y*block_size, z*block_size - settings.DEPTH/2]
                            # data.append(voxel_to_display)
                            data.append(voxel_to_display)
                            colors.append(total_voxels_color[idx][x, z, y])
    else:
        # Convert the labels to the new labels and assign them colors
        new_centres = [[], [], [], []]
        for label in total_labels[idx]:
            label = final_labels[label]
            new_centres[final_labels[label[0]]] = total_centers[idx][label[0]]
            if label == 0:
                colors.append([0, 0, 225])
            if label == 1:
                colors.append([0, 255, 0])
            if label == 2:
                colors.append([255, 0, 225])
            if label == 3:
                colors.append([255, 0, 0])
        final_centers.append(new_centres)
    
    # Create a mesh of the voxels
    if settings.CREATE_MESH:
        generate_mesh(total_voxel_volume_cleaned[idx], idx)
        print(f"Saved Mesh to voxel_mesh_frame_{idx}.png")

    return data, colors

def get_cam_positions():
    '''
    Returns the camera positions and colors for each camera

    :return: cameraposition2, colors
    '''
    
    cam_pos = np.zeros((4, 3, 1))
    for n_camera in range(1, settings.NUM_CAMERAS+1):
        # Get the camera parameters
        _, _, rvecs, tvecs = read_camera_parameters(n_camera)
        # Calculate the rotation matrix
        rotM = cv.Rodrigues(rvecs)[0]
        # Calculate the camera positions
        cam_pos[(n_camera-1)] = (-np.dot(np.transpose(rotM), tvecs / settings.GRID_TILE_SIZE))

    # Set the camera positions (and swap the y and z coordinates)
    camera_positions = [[cam_pos[0][0][0], -cam_pos[0][2][0], cam_pos[0][1][0]],
                       [cam_pos[1][0][0], -cam_pos[1][2][0], cam_pos[1][1][0]],
                       [cam_pos[2][0][0], -cam_pos[2][2][0], cam_pos[2][1][0]],
                       [cam_pos[3][0][0], -cam_pos[3][2][0], cam_pos[3][1][0]]]

    # Assign different colors to different cameras
    colors = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 0, 1.0]]

    return camera_positions, colors

def get_cam_rotation_matrices():
    '''
    Returns the camera rotation matrices for each camera

    :return: cam_rotations
    '''

    cam_rotations = []
    # Iterate over the cameras to retrieve the rotation matrices
    for n_camera in range(1, settings.NUM_CAMERAS+1):
        # Get the camera parameters
        _, _, rvecs, _ = read_camera_parameters(n_camera)
        # Calculate the rotation matrix
        rotM = cv.Rodrigues(rvecs)[0]
        # Post process the camera rotation matrix
        rotM = rotM.transpose()
        rotM = [rotM[0], rotM[2], rotM[1]]
        cam_rotations.append(glm.mat4(np.matrix(rotM).T))

    # For each rotation matrix, rotate it by -90 degrees around the y-axis
    for c in range(len(cam_rotations)):
        cam_rotations[c] = glm.rotate(cam_rotations[c], -np.pi/2 , [0, 1, 0])

    return cam_rotations

# Our extra functions

def increment_frame_count():
    '''
    Increments the frame count
    '''
    global frame_cnt
    global visualize_real
    if settings.VISUALIZE_REAL_COLORS == True:
        if visualize_real == False:
            frame_cnt += 1
        visualize_real = not visualize_real
    else:
        frame_cnt += 1

    return frame_cnt < min(settings.MAX_NUMBER_OF_FRAMES + len(settings.OFFLINE_IDX), settings.NUMBER_OF_FRAMES_TO_ANALYSE + len(settings.OFFLINE_IDX))

def decrement_frame_count():
    '''
    Decrements the frame count
    '''
    global frame_cnt
    global visualize_real
    if settings.VISUALIZE_REAL_COLORS == True:
        if visualize_real == True:
            frame_cnt -= 1
        visualize_real = not visualize_real
    else:
        frame_cnt -= 1

    return frame_cnt < min(settings.MAX_NUMBER_OF_FRAMES + len(settings.OFFLINE_IDX), settings.NUMBER_OF_FRAMES_TO_ANALYSE + len(settings.OFFLINE_IDX))

def get_frame_count():
    '''
    Returns the frame count

    :return: Frame count
    '''
    if frame_cnt < len(settings.OFFLINE_IDX):
        return -(frame_cnt+1)
    else:
        return frame_cnt - len(settings.OFFLINE_IDX) + settings.STARTING_FRAME_NUMBER

def create_lookup_table():
    '''
    Creates a look-up table to store the voxel coordinates and the corresponding pixel coordinates for each camera

    :param width: The width of the voxel grid
    :param height: The height of the voxel grid
    :param depth: The depth of the voxel grid

    :return: lookup_table
    '''
    width = settings.WIDTH
    height = settings.HEIGHT
    depth = settings.DEPTH

    lookup_table = []

    # Retrieve the variable from the pickle file if a lookup table has been already created
    if os.path.exists('lookup_table.pkl'):
        with open('lookup_table.pkl', 'rb') as f:
            return pickle.load(f)
    
    voxel_points = []
    
    print("\n\nSTEP 1 - Creating lookup table")
    # Iterate over the voxel volume to create the look-up table
    for x in tqdm(range(width), desc="Step 1.1 - Lookup Table Creation"):
        for y in range(height):
            for z in range(depth):
                # Calculate the voxel point for the opencv projectPoints function
                voxel_points.append([(x*block_size - width/2) * settings.GRID_TILE_SIZE, (z*block_size - depth/2) * settings.GRID_TILE_SIZE, -y*block_size * settings.GRID_TILE_SIZE])

    # Iterate over the cameras
    for c in tqdm(range(1, settings.NUM_CAMERAS+1), desc="Step 1.2 - Lookup Table - Projecting Points"):
        # Get the camera parameters
        camera_matrix, distortion_coeffs, rotation_vector, translation_vector = read_camera_parameters(c)
        # Project voxel point onto image plane of camera c
        img_points, _ = cv.projectPoints(np.array(voxel_points, np.float32), rotation_vector, translation_vector, camera_matrix, distortion_coeffs)

        lookup_table.append(img_points)

    # Save the variable to a pickle file
    with open('lookup_table.pkl', 'wb') as f:
        pickle.dump(lookup_table, f, protocol=4)

    return lookup_table

def create_voxel_model(lookup_table):
    '''
    Creates the voxel model

    :param lookup_table: The look-up table

    :return: total_voxel_volume, total_voxel_colors_per_cam, total_voxel_HSV_colors_per_cam
    '''

    if os.path.exists(f'voxel_models.pkl'):
        with open(f'voxel_models.pkl', 'rb') as f:
            return pickle.load(f)

    width = settings.WIDTH
    height = settings.HEIGHT
    depth = settings.DEPTH

    print("\n\nSTEP 2 - Creating voxel model")

    total_voxel_volume = []
    total_voxel_colors_per_cam = []
    total_voxel_HSV_colors_per_cam = []

    number_of_frames = get_lowest_frame_number()
    frame_step_size = int((number_of_frames - number_of_frames%settings.NUMBER_OF_FRAMES_TO_ANALYSE)/settings.NUMBER_OF_FRAMES_TO_ANALYSE)
    starting_frame = frame_step_size * settings.STARTING_FRAME_NUMBER
    frames = list(range(starting_frame, number_of_frames, frame_step_size))
    frames = settings.OFFLINE_IDX + frames
    print("Frames to analyse: ", frames)
    for i, frame_number in tqdm(enumerate(frames), desc=f"Creating Voxel Model - Frame Iteration"):
        if i >= settings.MAX_NUMBER_OF_FRAMES + len(settings.OFFLINE_IDX):
            continue
        print("Frame number: ", frame_number)
        foreground_mask = []
        color_images = []
        HSV_images = []
        # Create a voxel volume
        voxel_volume = np.ones((width, depth, height), dtype=bool)
        # Create a list to store the voxel coordinates and the corresponding colors
        voxel_colors_per_cam = np.zeros((width, depth, height, settings.NUM_CAMERAS, 3), dtype=int)
        voxel_HSV_colors_per_cam = np.zeros((width, depth, height, settings.NUM_CAMERAS, 3), dtype=int)
        # Iterate over the cameras to retrieve the foreground masks and color images for the current frame
        for n_camera in range(1, settings.NUM_CAMERAS+1):
            video_path = f'data/cam{n_camera}/foreground_mask.avi'
            cap = cv.VideoCapture(video_path)
            cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)
            _, frame = cap.read()
            frame_cvt = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            foreground_mask.append(frame_cvt) # cv.imread(f'data/cam{n_camera}/foreground_mask.jpg', 0)
            # foreground_mask.append(cv.imread(f'manual_masks/manual_mask_{n_camera}.jpg', 0))

            video_path_color = f'data/cam{n_camera}/video.avi'
            cap_color = cv.VideoCapture(video_path_color)
            cap_color.set(cv.CAP_PROP_POS_FRAMES, frame_number)
            _, frame_color = cap_color.read()
            color_images.append(frame_color)
            # Convert frame to HSV
            frame_hsv = cv.cvtColor(frame_color, cv.COLOR_BGR2HSV)
            HSV_images.append(frame_hsv)

            cap.release()
            cap_color.release()

            for x in tqdm(range(width), desc=f"Voxel Model - Voxel Projection - Camera {n_camera}/{settings.NUM_CAMERAS}", leave=False):
                for y in range(height):
                    for z in range(depth):
                        if not voxel_volume[x, z, y]:
                            continue
                        voxel_index = z + y * depth + x * (depth * height)

                        projection_x = -1 if np.isinf(lookup_table[n_camera-1][voxel_index][0][0]) else int(lookup_table[n_camera-1][voxel_index][0][0])
                        projection_y = -1 if np.isinf(lookup_table[n_camera-1][voxel_index][0][1]) else int(lookup_table[n_camera-1][voxel_index][0][1])
                        if projection_x < 0 or projection_y < 0 or projection_x >= foreground_mask[n_camera-1].shape[1] or projection_y >= foreground_mask[n_camera-1].shape[0] or not foreground_mask[n_camera-1][projection_y, projection_x]:
                            voxel_volume[x, z, y] = False
                            voxel_colors_per_cam[x, z, y, :] = [0, 0, 0]
                            voxel_HSV_colors_per_cam[x, z, y, :] = [0, 0, 0]
                        else:
                            voxel_colors_per_cam[x, z, y, n_camera-1] = color_images[n_camera-1][projection_y, projection_x, :]
                            voxel_HSV_colors_per_cam[x, z, y, n_camera-1] = HSV_images[n_camera-1][projection_y, projection_x, :]
        
        total_voxel_volume.append(voxel_volume)
        total_voxel_colors_per_cam.append(voxel_colors_per_cam)
        total_voxel_HSV_colors_per_cam.append(voxel_HSV_colors_per_cam)
    
    with open(f'voxel_models.pkl', 'wb') as f:
        data_to_save = (total_voxel_volume, total_voxel_colors_per_cam, total_voxel_HSV_colors_per_cam)
        pickle.dump(data_to_save, f, protocol=4)
    
    return total_voxel_volume, total_voxel_colors_per_cam, total_voxel_HSV_colors_per_cam

def generate_mesh(voxels, frame_number):
    '''
    Generates the mesh of the voxels

    :param voxels: The voxel volume
    :param frame_number: The frame number for the video

    :return: None (but saves the image to a file)
    '''

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

    # Set the viewing angle
    ax.set_xlim((-config['world_width']+100)/2, (config['world_width']+100)/2)
    ax.set_ylim((-config['world_height']+100)/2, (config['world_height']+100)/2)
    ax.set_zlim(0, config['world_depth'])

    plt.tight_layout()
    plt.savefig(f'voxel_mesh_frame_{frame_number}.png')

def create_background_model_gmm(video_path):
    '''
    Creates a background model using the Gaussian Mixture Model

    :param video_path: The path to the video

    :return: background_model
    '''
    if os.path.exists(f'{os.path.dirname(video_path)}/background_model.jpg'):
        print("Background model already created")
        return cv.imread(f'{os.path.dirname(video_path)}/background_model.jpg')

    # Open the video file
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening background video file")
        return None

    # Create the background subtractor object
    backSub = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    # Iterate over the frames to create the background model
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

def background_subtraction(frame, background_model_path, thresholds):
    '''
    Subtracts the background from the frame

    :param frame: The frame
    :param background_model_path: The path to the background model
    :param thresholds: The thresholds for the background subtraction

    :return: final_mask
    '''

    # Unpack the optimal thresholds
    h_thresh, s_thresh, v_thresh = thresholds
    # Load the background model and convert it to HSV
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

    # Large dilation to create a big mask around the subject
    kernel_1 = np.ones((5,5), np.uint8)
    dilation_mask = cv.dilate(threshold_mask, kernel_1, iterations=1)

    # Find contours for the blob algorithm
    contours_1, _ = cv.findContours(dilation_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area to identify blobs (large area to be sure to clean the whole image)
    min_blob_area_1 = 1500 
    blobs_1 = [cnt for cnt in contours_1 if cv.contourArea(cnt) > min_blob_area_1]

    # Draw the detected blobs
    blob_mask = np.zeros_like(dilation_mask)
    cv.drawContours(blob_mask, blobs_1, -1, (255), thickness=cv.FILLED)

    # Bitwise AND to get the final mask
    final_mask = cv.bitwise_and(threshold_mask, blob_mask)

    # Find contours for the blob algorithm
    contours_2, _ = cv.findContours(final_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area to identify blobs (large area to be sure to clean the whole image)
    min_blob_area_2 = 100
    blobs_2 = [cnt for cnt in contours_2 if cv.contourArea(cnt) > min_blob_area_2]

    # Draw the detected blobs
    blob_mask_2 = np.zeros_like(final_mask)
    cv.drawContours(blob_mask_2, blobs_2, -1, (255), thickness=cv.FILLED)

    final_mask = blob_mask_2

    return final_mask

def manual_segmentation_comparison(camera_number, first_video_frame, background_model_path, manual_mask_path, steps=[50, 10, 5, 1]):
    '''
    Finds the optimal thresholds for the background subtraction using the manual mask

    :param first_video_frame: The first frame of the video
    :param background_model_path: The path to the background model
    :param manual_mask_path: The path to the manual mask
    :param steps: The sizes of the steps for the background subtraction

    :return: optimal_thresholds
    '''
    if os.path.exists(f'data/cam{camera_number}/thresholds.xml'):
        print("Thresholds already found")
        # Open a FileStorage object to read data from the XML file
        fs = cv.FileStorage(f'data/cam{camera_number}/thresholds.xml', cv.FILE_STORAGE_READ)
        # Read the optimal thresholds from the file
        optimal_thresholds = fs.getNode("Optimal-Thresholds").mat().ravel().astype(int).tolist()
        fs.release()

        print(f'Optimal thresholds: Hue={optimal_thresholds[0]}, Saturation={optimal_thresholds[1]}, Value={optimal_thresholds[2]}')

        return optimal_thresholds

    # Initialize the optimal thresholds and score
    optimal_thresholds = None
    optimal_score = float('inf')
    previous_step = 255
    optimal_thresholds = (0, 0, 0)

    # Nested loops for searching threshold values with increasingly small steps
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
                    # Apply the background subtraction with the current thresholds
                    test_threshold = (h_thresh, s_thresh, v_thresh)
                    segmented = background_subtraction(first_video_frame, background_model_path, test_threshold)
                    xor_result = cv.bitwise_xor(segmented, cv.imread(manual_mask_path, 0))
                    # Assign the score based on the number of non-zero pixels in the XOR result
                    score = cv.countNonZero(xor_result)
                    
                    # Update the optimal thresholds if the current score is better
                    if score < optimal_score:
                        optimal_score = score
                        optimal_thresholds = (h_thresh, s_thresh, v_thresh)

        # Print the optimal thresholds found in this iteration
        print(f"Optimal thresholds after refinement with step {current_step}: Hue={optimal_thresholds[0]}, Saturation={optimal_thresholds[1]}, Value={optimal_thresholds[2]}")

        # Decrease step size for the next iteration
        previous_step = current_step

    # Print the final optimal thresholds
    print(f'Optimal thresholds: Hue={optimal_thresholds[0]}, Saturation={optimal_thresholds[1]}, Value={optimal_thresholds[2]}')

    directory = f'data/cam{camera_number}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Define the file path for the config file
    file_name = f"{directory}/thresholds.xml"
    
    # Open a FileStorage object to write data to the XML file
    fs = cv.FileStorage(file_name, cv.FILE_STORAGE_WRITE)

    fs.write("Optimal-Thresholds", np.asarray(optimal_thresholds, dtype=np.float32))

    fs.release()
    
    return optimal_thresholds

def create_segmented_video(video_path, background_model_path, optimal_thresholds):
    '''
    Creates a video with the segmented frames

    :param video_path: The path to the video
    :param background_model_path: The path to the background model
    :param optimal_thresholds: The optimal thresholds for the background subtraction

    :return: None
    '''
    
    # Create the output video file path
    folder_path = os.path.dirname(video_path)
    output_video_path = f'{folder_path}/foreground_mask.avi'  # Update with the desired output video file path
    if os.path.exists(output_video_path):
        print("Segmented video already created")
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
    out = cv.VideoWriter(output_video_path, fourcc, fps, size, isColor=False)

    # Iterate over the frames to create the segmented video
    while True:
        
        ret, frame = cap.read()
        if not ret:
            break

        # Apply the background subtraction to each frame
        segmented = background_subtraction(frame, background_model_path, optimal_thresholds)

        # Apply dilation to fill in gaps
        kernel_2 = np.ones((2, 2), np.uint8)
        dilation_mask_2 = cv.dilate(segmented, kernel_2, iterations=1)

        # Find contours for the blob mask
        contours_2, _ = cv.findContours(dilation_mask_2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area to identify blobs
        min_blob_area_2 = 20  # Adjust this threshold as needed
        blobs_2 = [cnt for cnt in contours_2 if cv.contourArea(cnt) > min_blob_area_2]

        # Draw the detected blobs
        blob_mask_2 = np.zeros_like(dilation_mask_2)
        cv.drawContours(blob_mask_2, blobs_2, -1, (255), thickness=cv.FILLED)

        segmented = blob_mask_2

        # Write the segmented frame to the output video
        out.write(segmented)
    
    cap.release()
    out.release()
    cv.destroyAllWindows()

def read_camera_parameters(camera_number):
    '''
    Reads the camera parameters from the XML file

    :param camera_number: The camera number

    :return: camera_matrix, dist_coeffs, rvecs, tvecs
    '''
    # Set the file name
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

def ray_trace(total_voxel_volume, total_voxel_colors_per_cam, total_voxel_HSV_colors_per_cam):
    """
    Determines which voxels are visible from the camera position and assigns colors to the visible voxels.

    :param total_voxel_volume: The voxel volume
    :param total_voxel_colors_per_cam: The colors of the voxels
    :param total_voxel_HSV_colors_per_cam: The HSV colors of the voxels

    :return: total_visible_voxels_per_cam, total_visible_voxels_colors_per_cam
    """
    if os.path.exists(f'ray_trace.pkl'):
        with open(f'ray_trace.pkl', 'rb') as f:
            return pickle.load(f)
    
    width = settings.WIDTH
    height = settings.HEIGHT
    depth = settings.DEPTH

    print("\n\nSTEP 4 - Assigning colors")

    total_voxels_color = []
    total_visible_voxels_per_cam = []
    total_visible_voxels_colors_per_cam = []

    number_of_frames = get_lowest_frame_number()
    frame_step_size = int((number_of_frames - number_of_frames%settings.NUMBER_OF_FRAMES_TO_ANALYSE)/settings.NUMBER_OF_FRAMES_TO_ANALYSE)
    starting_frame = frame_step_size * settings.STARTING_FRAME_NUMBER
    frames = list(range(starting_frame, number_of_frames, frame_step_size))
    frames = settings.OFFLINE_IDX + frames
    for idx, frame_number in tqdm(enumerate(frames), desc="Assigning Colors - Frame Iteration"):
        if idx >= settings.MAX_NUMBER_OF_FRAMES + len(settings.OFFLINE_IDX):
            continue
        # Get the voxel positions
        voxel_positions = np.argwhere(total_voxel_volume[idx])
        voxel_positions[1], voxel_positions[2] = voxel_positions[2], voxel_positions[1]
        # Create a copy of the voxels array to store the colors
        voxels_color = np.empty((total_voxel_volume[idx].shape[0], total_voxel_volume[idx].shape[1], total_voxel_volume[idx].shape[2]), dtype=object)
        total_voxel_colors_per_cam[idx][1], total_voxel_colors_per_cam[idx][2] = total_voxel_colors_per_cam[idx][2], total_voxel_colors_per_cam[idx][1]
        # Get the camera positions
        camera_position, _ = get_cam_positions()

        visible_voxels_per_cam = np.zeros((settings.NUM_CAMERAS, width, depth, height), dtype=bool)
        visible_voxels_colors_per_cam = np.zeros((settings.NUM_CAMERAS, width, depth, height, 3), dtype=np.float32)

        for n_camera in tqdm(range(1, settings.NUM_CAMERAS+1), desc="Color - Ray Tracing", leave=False):
            voxel_grid = total_voxel_volume[idx].copy()
            voxel_grid[1], voxel_grid[2] = voxel_grid[2], voxel_grid[1]
            voxel_colored = total_voxel_volume[idx].copy()
            voxel_colored[1], voxel_colored[2] = voxel_colored[2], voxel_colored[1]
            camera_position[n_camera-1] = [camera_position[n_camera-1][0] + width/2, camera_position[n_camera-1][2] + depth/2, camera_position[n_camera-1][1]]
            for target_voxel in tqdm(voxel_positions, desc=f"Color - Ray Tracing - Camera {n_camera}", leave=False):
                if voxel_colored[target_voxel[0], target_voxel[1], target_voxel[2]]:
                    # Calculate the direction vector from the camera to the target voxel
                    direction = target_voxel - camera_position[n_camera-1]
                    distance_to_target = np.linalg.norm(direction)
                    direction /= distance_to_target

                    # Step size for the ray
                    step_size = 1.0

                    # Flag: True if the current voxel is the first voxel along the ray
                    is_first = True
                    # Flag: True if the ray is outside the voxel grid
                    outside = True

                    step = 1
                    while True:
                        # Calculate the point along the ray at the current step
                        point_along_ray = camera_position[n_camera-1] + direction * step * step_size
                        step += 1
                        # Round the point to the nearest voxel
                        x, y, z = np.round(point_along_ray).astype(int)
                        # print("Point along ray: ", point_along_ray)
                        
                        # Check if we're outside the bounds of the voxel grid
                        if x < 0 or x >= voxel_grid.shape[0] or y < 0 or y >= voxel_grid.shape[1] or z < 0 or z >= voxel_grid.shape[2]:
                            # Break the loop if we're outside the voxel grid
                            if outside == False:
                                break
                            continue
                        
                        outside = False
                        
                        # Check if the current voxel is on
                        if voxel_colored[x, y, z] or (voxel_grid[x, y, z] and is_first):
                            # If the current voxel is the first visible voxel along the ray, keeps it in the visible ones
                            if is_first:
                                voxel_colored[x, y, z] = True
                                if not voxels_color[x, y, z]:
                                    voxels_color[x, y, z] = []
                                voxels_color[x, y, z].append(total_voxel_colors_per_cam[idx][target_voxel[0], target_voxel[1], target_voxel[2], n_camera-1]/255)
                                visible_voxels_per_cam[n_camera-1, x, y, z] = True
                                visible_voxels_colors_per_cam[n_camera-1, x, y, z] = total_voxel_HSV_colors_per_cam[idx][target_voxel[0], target_voxel[1], target_voxel[2], n_camera-1]
                                is_first = False
                            else:
                                # If the current voxel is not the first visible voxel along the ray, then it is occluded
                                voxel_colored[x, y, z] = False
        reshape_voxel_color = np.array([[[np.mean(lst, axis=0) if lst is not None else [0, 0, 0] for lst in row] for row in subarray] for subarray in voxels_color])
        total_voxels_color.append(reshape_voxel_color)
        total_visible_voxels_per_cam.append(visible_voxels_per_cam)
        total_visible_voxels_colors_per_cam.append(visible_voxels_colors_per_cam)
    
    with open(f'ray_trace.pkl', 'wb') as f:
        data_to_save = (total_voxels_color, total_visible_voxels_per_cam, total_visible_voxels_colors_per_cam)
        pickle.dump(data_to_save, f, protocol=4)

    return total_voxels_color, total_visible_voxels_per_cam, total_visible_voxels_colors_per_cam

def create_all_models():
    '''
    Creates all the models needed for the assignment

    :return: total_voxels_color, total_visible_voxels_per_cam, total_visible_voxels_colors_per_cam, total_voxel_volume_cleaned, total_centers, total_labels, total_voxels
    '''

    lookup_table = create_lookup_table()

    print("\nLookup table created")

    total_voxel_volume, total_voxel_colors_per_cam, total_voxel_HSV_colors_per_cam = create_voxel_model(lookup_table)

    print("\nVoxel model created")

    if os.path.exists(f'post_clustering.pkl'):
        with open(f'post_clustering.pkl', 'rb') as f:
            total_labels, total_centers, total_voxels, total_voxel_volume_cleaned = pickle.load(f)
    else:
        print("\n\nSTEP 3 - New Voxels Projection")
        total_labels = []
        total_centers = []
        total_voxels = []
        total_voxel_volume_cleaned = []

        for idx in tqdm(range(len(total_voxel_volume)), desc="New Voxels Projection - Frame Iteration"):
            voxels = []
            # Create a counter to store the number of visible voxels in all cameras
            visible_all_cameras = 0
            # Iterate over voxels to store the visible ones
            for x in tqdm(range(settings.WIDTH), desc="Voxel Projection - Visible Voxels", leave=False):
                for y in range(settings.HEIGHT):
                    for z in range(settings.DEPTH):
                        if total_voxel_volume[idx][x, z, y]:
                            visible_all_cameras += 1
                            # Store the voxel coordinates to output
                            voxel_to_display = [x*block_size - settings.WIDTH/2, y*block_size, z*block_size - settings.DEPTH/2]
                            # data.append(voxel_to_display)
                            voxels.append(voxel_to_display)
                            
                            # colors.append(total_voxels_color[idx][x, z, y])
            
            labels, centers, voxels = col_cl.cluster_voxels(voxels)

            voxel_volume_cleaned = np.zeros((settings.WIDTH, settings.DEPTH, settings.HEIGHT), dtype=bool)
            for voxel in voxels:
                voxel_volume_cleaned[int((voxel[0]+settings.WIDTH/2)/block_size)][int((voxel[2]+settings.DEPTH/2)/block_size)][int(voxel[1]/block_size)] = True

            total_labels.append(labels)
            total_centers.append(centers)
            total_voxels.append(voxels)
            total_voxel_volume_cleaned.append(voxel_volume_cleaned)
    
        with open(f'post_clustering.pkl', 'wb') as f:
            data_to_save = (total_labels, total_centers, total_voxels, total_voxel_volume_cleaned)
            pickle.dump(data_to_save, f, protocol=4)
    
    print("\nNew Voxels projected")
    
    total_voxels_color, total_visible_voxels_per_cam, total_visible_voxels_colors_per_cam = ray_trace(total_voxel_volume_cleaned, total_voxel_colors_per_cam, total_voxel_HSV_colors_per_cam)

    print("\nColors assigned")

    total_labels, total_centers, total_voxels, total_voxel_volume_cleaned = col_cl.remove_ghost_voxels(total_labels, total_voxels, total_visible_voxels_per_cam)

    print("\n\nAll Frames are ready to be displayed: press 'g' to visualise the next frame")

    return total_voxels_color, total_visible_voxels_per_cam, total_visible_voxels_colors_per_cam, total_voxel_volume_cleaned, total_centers, total_labels, total_voxels

def get_lowest_frame_number():
    '''
    Gets the lowest frame number from all the cameras

    :return: lowest_frame_number
    '''
    lowest_frame_number = float('inf')
    for n_camera in range(1, settings.NUM_CAMERAS+1):
        video_path = f'data/cam{n_camera}/foreground_mask.avi'
        cap = cv.VideoCapture(video_path)
        frame_number = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        if frame_number < lowest_frame_number:
            lowest_frame_number = frame_number
        cap.release()

    return lowest_frame_number

def draw_centers_graph():
    '''
    Draws the centers graph

    :return: None
    '''

    # Extract x and y coordinates from each array
    x_points = [[], [], [], []]
    y_points = [[], [], [], []]
    colors = ['g', 'b', 'r', 'm']  # Colors for each position in the array

    if os.path.exists(f'centers_graph.pkl'):
        with open(f'centers_graph_2.pkl', 'rb') as f:
            x_points, y_points = pickle.load(f)

    for j, arr in enumerate(final_centers):
        for i, point in enumerate(arr):
            if j >= settings.NUM_CAMERAS:
                x_points[i].append(point[0])
                y_points[i].append(point[1])
    
    if not os.path.exists(f'centers_graph.pkl'):
        with open(f'centers_graph_2.pkl', 'wb') as f:
            data_to_save = (x_points, y_points)
            pickle.dump(data_to_save, f, protocol=4)

    # Plot the points
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Points Plot')
    plt.legend()
    plt.grid(True)
    new_x_points = x_points[:50] + x_points[55:]
    new_y_points = y_points[:50] + y_points[55:]
    for i in range(4):
        plt.plot(new_x_points[i], new_y_points[i], color=colors[i], label=f'Position {i+1}')
    plt.savefig('centers_graph.png')
    plt.close()

    # Plot the points for each variable separately
    for i in range(4):
        plt.figure() 
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title(f'Points Plot for Position {i+1}')
        plt.grid(True)
        plt.plot(new_x_points[i], new_y_points[i], color=colors[i])
        plt.xlim(-100, 40)
        plt.ylim(-70, 70)
        plt.savefig(f'position_{i+1}_graph.png')
        plt.close()

    return

# Offline preparatory part

# Get camera intrinsics and extrinsics, create the background model and the segmented video
for camera_number in range(1, settings.NUM_CAMERAS+1):
    print(f"\n\nProcessing camera {camera_number}")
    # Analise the video and gets the camera intrinsics and extrinsics
    cc.get_camera_intrinsics_and_extrinsics(camera_number)
    # Create the background model
    background_video_path = f'data/cam{camera_number}/background.avi'
    background_model = create_background_model_gmm(background_video_path)
    # Create the segmented video
    manual_mask_path = f'manual_masks/manual_mask_{camera_number}.jpg'
    video_path = f'data/cam{camera_number}/video.avi'
    _, first_video_frame = cv.VideoCapture(video_path).read()
    background_model_path = f'data/cam{camera_number}/background_model.jpg'
    optimal_thresholds = manual_segmentation_comparison(camera_number, first_video_frame, background_model_path, manual_mask_path, steps=[50, 10, 5, 1])
    create_segmented_video(video_path, background_model_path, optimal_thresholds)
# Creates and store all the data necessary for the color matching
total_voxels_color, total_visible_voxels_per_cam, total_visible_voxels_colors_per_cam, total_voxel_volume_cleaned, total_centers, total_labels, total_voxels = create_all_models()
cam_color_models_offline, _ = col_cl.color_model(total_voxels, total_labels, total_visible_voxels_per_cam, total_visible_voxels_colors_per_cam, offline = True)