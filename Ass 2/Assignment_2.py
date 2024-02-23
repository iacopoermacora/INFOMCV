import camera_calibration as cc
import settings as settings
import cv2 as cv
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time

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
    cv.imwrite(f'data/cam{camera_number}/background_model.jpg', background_model)
    return background_model

def background_subtraction(video_path, background_model_path, h_thresh, s_thresh, v_thresh, thresh_search=False):
    # Load the background model
    background_model = cv.imread(background_model_path)
    background_model_hsv = cv.cvtColor(background_model, cv.COLOR_BGR2HSV)
    # Open the video file
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to HSV
        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # Calculate the absolute difference
        diff = cv.absdiff(frame_hsv, background_model_hsv)

       # Threshold for each channel
        _, thresh_h = cv.threshold(diff[:,:,0], h_thresh, 255, cv.THRESH_BINARY) 
        _, thresh_s = cv.threshold(diff[:,:,1], s_thresh, 255, cv.THRESH_BINARY) 
        _, thresh_v = cv.threshold(diff[:,:,2], v_thresh, 255, cv.THRESH_BINARY)


        # Combine the thresholds 
        combined_mask = cv.bitwise_and(thresh_v, cv.bitwise_and(thresh_h, thresh_s))

        if thresh_search == True:
            break

    if thresh_search == False:
        cv.imshow('Foreground Mask', combined_mask)
        cv.waitKey(0) 

    cap.release()
    cv.destroyAllWindows()

    return combined_mask

def manual_segmentation_comparison(video_path, background_model_path, manual_mask_path, steps=[50, 10, 5, 1]):
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
                    segmented = background_subtraction(video_path, background_model_path, h_thresh, s_thresh, v_thresh, thresh_search=True)
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
    segmented = background_subtraction(video_path, background_model_path, optimal_thresholds[0], optimal_thresholds[1], optimal_thresholds[2])

    return optimal_thresholds

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

# Step 2: Construct Lookup Table
def construct_lookup_table(voxel_volume, calibration_data, num_views):
    lookup_table = []

    for xv in range(voxel_volume.shape[0]):
        for yv in range(voxel_volume.shape[1]):
            for zv in range(voxel_volume.shape[2]):
                voxel_point = np.array([[xv, yv, zv]], dtype=np.float32)

                for c in range(num_views):
                    camera_matrix, distortion_coeffs, rotation_vector, translation_vector = calibration_data[c]
                    # Project voxel point onto image plane of camera c
                    img_point, _ = cv.projectPoints(voxel_point, rotation_vector, translation_vector,
                                                      camera_matrix, distortion_coeffs)
                    img_point = tuple(img_point[0].ravel())
                    lookup_table.append(((xv, yv, zv), c, img_point))

    return lookup_table
    
# Call the function to get the camera intrinsics and extrinsics for each camera
for camera_number in range(1, settings.num_cameras+1):
    cc.get_camera_intrinsics_and_extrinsics(camera_number)
    # background model
    background_video_path = f'data/cam{camera_number}/background.avi'
    background_model = create_background_model_gmm(background_video_path)
    
    # display of the background model
    if background_model is not None:
        cv.imshow('Background Model', background_model)
        cv.waitKey(0) 
        cv.destroyAllWindows()
    else:
        print("Failed to create background model")
    
    # manual subtraction
    manual_mask_path = f'data/cam{camera_number}/manual_mask.jpg'   
    video_path = f'data/cam{camera_number}/video.avi'
    background_model_path = f'data/cam{camera_number}/background_model.jpg'
    optimal_thresholds = manual_segmentation_comparison(video_path, background_model_path, manual_mask_path, steps=[50, 10, 5, 1])

