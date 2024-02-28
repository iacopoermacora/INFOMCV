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

    data, colors = [], []

    # Create a voxel volume
    voxel_volume = np.zeros((width, height, depth, settings.NUM_CAMERAS), dtype=bool)
    # Create a list to store the voxel coordinates and the corresponding colors
    voxel_to_colors = np.zeros((width, height, depth, settings.NUM_CAMERAS, 3), dtype=int)
    # Create a lookup table to store the voxel coordinates and the corresponding pixel coordinates for each camera
    lookup_table = create_lookup_table(voxel_volume)

    # Get the frame model
    data, colors = get_frame_model(lookup_table, voxel_volume, voxel_to_colors, width, height, depth)

    return data, colors

def get_frame_model(lookup_table, voxel_volume, voxel_to_colors, width, height, depth):
    '''
    Generates the frame model for the 3D projection

    :param lookup_table: The lookup table
    :param foreground_mask: The foreground mask
    :param color_images: The color images
    :param voxel_volume: The voxel volume
    :param voxel_to_colors: The voxel to colors array
    :param width: The width of the voxel grid
    :param height: The height of the voxel grid
    :param depth: The depth of the voxel grid

    :return: data, colors
    '''
    maximum_frame_number = 0
    for n_camera in range(1, settings.NUM_CAMERAS+1):
        video_path = f'data/cam{n_camera}/foreground_mask.avi'
        cap = cv.VideoCapture(video_path)
        frame_number_cam = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        if frame_number_cam > maximum_frame_number:
            maximum_frame_number = frame_number_cam
        

    data_list = []
    colors_list = []
    
    for frame_n in tqdm(range(0, maximum_frame_number), desc="Frame Progress in voxel model"):
        # Create a list to store the voxel coordinates and the corresponding colors
        data = []
        colors = []

        foreground_mask = []
        color_images = []

        # Iterate over the cameras to retrieve the foreground masks and color images for the current frame
        for n_camera in range(1, settings.NUM_CAMERAS+1):
            video_path = f'data/cam{n_camera}/foreground_mask.avi'
            cap = cv.VideoCapture(video_path)
            cap.set(cv.CAP_PROP_POS_FRAMES, frame_n)
            _, frame = cap.read()
            frame_cvt = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            foreground_mask.append(frame_cvt) # cv.imread(f'data/cam{n_camera}/foreground_mask.jpg', 0)

            video_path_color = f'data/cam{n_camera}/video.avi'
            cap_color = cv.VideoCapture(video_path_color)
            cap_color.set(cv.CAP_PROP_POS_FRAMES, frame_n)
            _, frame_color = cap_color.read()
            color_images.append(frame_color)
            cap.release()
            cap_color.release()
        
        foreground_mask_2 = foreground_mask.copy()

        if not frame_n == 0:
            for n_camera in range(1, settings.NUM_CAMERAS+1):
                foreground_mask_2[n_camera-1] = cv.bitwise_or(foreground_mask[n_camera-1], previous_foreground_mask[n_camera-1])
                cv.imshow(f"Foreground Mask {n_camera}", foreground_mask_2[n_camera-1])
                cv.waitKey(0)
                cv.destroyAllWindows()
                cv.waitKey(1)
            voxel_volume = previous_voxel_volume
            voxel_to_colors = previous_voxel_to_colors
    
        # Iterate over pixels in the lookup table
        for voxel_coords, n_camera, pixel_coords in lookup_table:
            # Check if pixel is foreground in the corresponding view
            if is_foreground(pixel_coords, foreground_mask_2[n_camera-1]):
                if not frame_n == 0:
                    if voxel_volume[voxel_coords[0], voxel_coords[1], voxel_coords[2], (n_camera-1)] == True:
                        voxel_volume[voxel_coords[0], voxel_coords[1], voxel_coords[2], (n_camera-1)] = False
                        voxel_to_colors[voxel_coords[0], voxel_coords[1], voxel_coords[2], (n_camera-1)] = (0, 0, 0)
                    else:
                        voxel_volume[voxel_coords[0], voxel_coords[1], voxel_coords[2], (n_camera-1)] = True
                        # Store the color of the voxel in the corresponding camera
                        color = color_images[n_camera-1][int(pixel_coords[0]), int(pixel_coords[1]), :]
                        # Store the color in the voxel_to_colors array
                        voxel_to_colors[voxel_coords[0], voxel_coords[1], voxel_coords[2], (n_camera-1)] = color
                else:
                    voxel_volume[voxel_coords[0], voxel_coords[1], voxel_coords[2], (n_camera-1)] = True
                    # Store the color of the voxel in the corresponding camera
                    color = color_images[n_camera-1][int(pixel_coords[0]), int(pixel_coords[1]), :]
                    # Store the color in the voxel_to_colors array
                    voxel_to_colors[voxel_coords[0], voxel_coords[1], voxel_coords[2], (n_camera-1)] = color
                

        visible_all_cameras = 0
        # Iterate over voxels to mark them visible if visible in all views
        voxels = np.all(voxel_volume, axis=3)
        # Set the voxel color visibility array to store the visibility of each voxel in each camera
        voxel_color_visibility = np.zeros((settings.NUM_CAMERAS, width, height, depth), dtype=bool)
        for n_camera in range(1, settings.NUM_CAMERAS+1):
            # Get the camera positions and swap the y and z coordinates
            camera_position, _ = get_cam_positions()
            camera_position[n_camera-1][1], camera_position[n_camera-1][2] = camera_position[n_camera-1][2], camera_position[n_camera-1][1]
            # Get the voxel positions
            voxel_positions = np.argwhere(voxels)
            # Create a copy of the voxels array to store the colors
            voxels_for_colors = voxels.copy()
            # Set the voxel color visibility array to store the visibility of each voxel in each camera
            voxel_color_visibility[n_camera-1] = ray_trace(camera_position[n_camera-1], voxel_positions, voxels_for_colors)
        
        # Iterate over voxels to store the visible ones
        for x in range(width):
            for y in range(height):
                for z in range(depth):
                    if voxels[x, y, z]:
                        visible_all_cameras += 1
                        # Store the voxel coordinates to output
                        data.append([(x*block_size - width/2) - settings.GRID_TILE_SIZE/2, (z*block_size), (y*block_size - depth/2) - settings.GRID_TILE_SIZE/2])
                        color_voxel_cam = []
                        # Iterate over the cameras to store the colors if the voxel is visible on that camera
                        for n_camera in range(1, settings.NUM_CAMERAS+1):
                            if voxel_color_visibility[n_camera-1, x, y, z]:
                                color_voxel_cam.append(voxel_to_colors[x, y, z, n_camera-1] / 255)
                        
                        # If the voxel is visible in at least one camera, store the average color otherwise store a shade of grey
                        if len(color_voxel_cam) > 0:
                            color_voxel = np.mean(color_voxel_cam, axis=0)
                        else:
                            color_voxel = [0, 0, 0] # TODO: Change to some shade of grey
                        colors.append(color_voxel)

        previous_foreground_mask = foreground_mask.copy()
        previous_voxel_volume = voxel_volume.copy()
        previous_voxel_to_colors = voxel_to_colors.copy()
        
        print(f"Total voxels visible in all cameras: {visible_all_cameras}")

        # Create a mesh of the voxels
        generate_mesh(voxels, frame_n)
        print(f"Saved Mesh to voxel_mesh_frame_{frame_n}.png")

        data_list.append(data)
        colors_list.append(colors)
    
    return data_list[settings.FRAME_NUMBER], colors_list[settings.FRAME_NUMBER]


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

def create_lookup_table(voxel_volume):
    '''
    Creates a look-up table to store the voxel coordinates and the corresponding pixel coordinates for each camera

    :param voxel_volume: The voxel volume

    :return: lookup_table
    '''

    lookup_table = []

    # Retrieve the variable from the pickle file if a lookup table has been already created
    if os.path.exists('lookup_table.pkl'):
        with open('lookup_table.pkl', 'rb') as f:
            return pickle.load(f)
    
    # Get the dimensions of the background model
    (heightImg, widthImg) = cv.imread(f'data/cam1/background_model.jpg', 0).shape
    
    # Iterate over the voxel volume to create the look-up table
    for x in tqdm(range(voxel_volume.shape[0]), desc="Lookup Table Progress"):
        for y in range(voxel_volume.shape[1]):
            for z in range(voxel_volume.shape[2]):
                # Calculate the voxel point for the opencv projectPoints function
                voxel_point = np.array([[(x*block_size - voxel_volume.shape[0]/2) * settings.GRID_TILE_SIZE, (y*block_size - voxel_volume.shape[1]/2) * settings.GRID_TILE_SIZE, -z*block_size * settings.GRID_TILE_SIZE]], dtype=np.float32)

                # Iterate over the cameras
                for c in range(1, settings.NUM_CAMERAS+1):
                    # Get the camera parameters
                    camera_matrix, distortion_coeffs, rotation_vector, translation_vector = read_camera_parameters(c)
                    # Project voxel point onto image plane of camera c
                    img_point, _ = cv.projectPoints(voxel_point, rotation_vector, translation_vector, camera_matrix, distortion_coeffs)
                    img_point = np.reshape(img_point, 2)
                    img_point = img_point[::-1]
                    

                    # Only accept pixels inside the image
                    if 0 <= img_point[0] < heightImg and 0 <= img_point[1] < widthImg:
                        # Store {XV, YV, ZV}, c and {xim, yim} in the look-up table
                        lookup_table.append((((x, y, z), c, img_point)))

    # Save the variable to a pickle file
    with open('lookup_table.pkl', 'wb') as f:
        pickle.dump(lookup_table, f, protocol=4)

    return lookup_table

def is_foreground(pixel_coords, foreground_mask):
    '''
    Checks if a pixel is foreground in the corresponding view

    :param pixel_coords: The pixel coordinates
    :param foreground_mask: The foreground mask

    :return: True if the pixel is foreground, False otherwise
    '''

    # Check if pixel is foreground in the corresponding view
    y, x = int(pixel_coords[0]), int(pixel_coords[1]) # OpenCV uses (y, x) indexing

    return foreground_mask[y, x] == 255

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
    ax.set_xlim((-config['world_width']+100)/2, (config['world_width']+100)/2) # TODO: Fix these according to the last chosen size of the voxel grid
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

    return final_mask

def manual_segmentation_comparison(first_video_frame, background_model_path, manual_mask_path, steps=[50, 10, 5, 1]):
    '''
    Finds the optimal thresholds for the background subtraction using the manual mask

    :param first_video_frame: The first frame of the video
    :param background_model_path: The path to the background model
    :param manual_mask_path: The path to the manual mask
    :param steps: The sizes of the steps for the background subtraction

    :return: optimal_thresholds
    '''
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
                    segmented = background_subtraction(first_video_frame, background_model_path, test_threshold) # TODO: Change to frame
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

def ray_trace(camera_position, voxel_positions, voxel_grid):
    """
    Determines which voxels are visible from the camera position.

    :param camera_position: The (x, y, z) coordinates of the camera.
    :param voxel_positions: A list of (x, y, z) coordinates for each voxel.
    :param voxel_grid: An array representing the presence (True) or absence (False) of voxels.
    :return: A list of booleans indicating visibility for each voxel in voxel_positions.
    """

    for target_voxel in voxel_positions:
        if voxel_grid[target_voxel[0], target_voxel[1], target_voxel[2]]:
            # Calculate the direction vector from the camera to the target voxel
            direction = target_voxel - camera_position
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
                point_along_ray = camera_position + direction * step * step_size
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
                if voxel_grid[x, y, z]:
                    # If the current voxel is the first visible voxel along the ray, keeps it in the visible ones
                    if is_first:
                        is_first = False
                    else:
                        # If the current voxel is not the first visible voxel along the ray, then it is occluded
                        voxel_grid[x, y, z] = False

    return voxel_grid


# Call the function to get the camera intrinsics and extrinsics for each camera
for camera_number in range(1, settings.NUM_CAMERAS+1):
    # Analise the video and gets the camera intrinsics and extrinsics
    cc.get_camera_intrinsics_and_extrinsics(camera_number)
    # Create the background model
    background_video_path = f'data/cam{camera_number}/background.avi'
    background_model = create_background_model_gmm(background_video_path)
    # Create the segmented video
    manual_mask_path = f'data/cam{camera_number}/manual_mask.jpg'
    video_path = f'data/cam{camera_number}/video.avi'
    _, first_video_frame = cv.VideoCapture(video_path).read()
    background_model_path = f'data/cam{camera_number}/background_model.jpg'
    optimal_thresholds = manual_segmentation_comparison(first_video_frame, background_model_path, manual_mask_path, steps=[50, 10, 5, 1])
    create_segmented_video(video_path, background_model_path, optimal_thresholds)
    