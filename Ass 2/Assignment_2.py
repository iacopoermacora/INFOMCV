import camera_calibration as cc
import settings as settings
import cv2 as cv
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time

def background_subtraction():
    cap = cv.VideoCapture('vtest.avi')
    fgbg = cv.bgsegm.createBackgroundSubtractorMOG()
    while(1):
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)
        cv.imshow('frame',fgmask)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv.destroyAllWindows()

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

'''def manual_segmentation_comparison(video_path, background_model_path, manual_mask_path, threshold_ranges):
    # Load the manual segmentation mask
    manual_mask = cv.imread(manual_mask_path, 0)

    # Initialize variables for the optimal thresholds and their score
    optimal_thresholds = None
    optimal_score = float('inf')

    # Initialize containers for scores
    hue_scores = np.zeros(256)
    saturation_scores = np.zeros(256)
    value_scores = np.zeros(256)
    counts_hue = np.zeros(256)
    counts_saturation = np.zeros(256)
    counts_value = np.zeros(256)

    hue_counter = 0
    saturation_counter = 0
    value_counter = 0

    # Iterate over the range of thresholds for each channel
    for h_thresh in tqdm(threshold_ranges['hue'], desc="hue"):
        saturation_counter = 0
        for s_thresh in tqdm(threshold_ranges['saturation'], desc="saturation", leave=False):
            value_counter = 0
            for v_thresh in tqdm(threshold_ranges['value'], desc="value", leave=False):
                # Segment the frame using the current set of thresholds
                segmented = background_subtraction(video_path, background_model_path, h_thresh, s_thresh, v_thresh, thresh_search=True)

                # Compare the segmentation with the manual mask using XOR
                xor_result = cv.bitwise_xor(segmented, manual_mask)
                score = cv.countNonZero(xor_result)

                # Accumulate scores
                hue_scores[hue_counter] += score
                saturation_scores[saturation_counter] += score
                value_scores[value_counter] += score
                counts_hue[hue_counter] += 1
                counts_saturation[saturation_counter] += 1
                counts_value[value_counter] += 1

                # Update optimal thresholds if current score is better (lower)
                if score < optimal_score:
                    optimal_score = score
                    optimal_thresholds = (h_thresh, s_thresh, v_thresh)

                value_counter +=1
            saturation_counter += 1
        hue_counter += 1
    
    # Calculate average scores
    print("hue scores", hue_scores)
    print("sat scores", saturation_scores)
    print("val scores", value_scores)
    print("hue counts", counts_hue)
    print("sat counts", counts_saturation)
    print("val counts", counts_value)
    avg_hue_scores = hue_scores / counts_hue
    avg_saturation_scores = saturation_scores / counts_saturation
    avg_value_scores = value_scores / counts_value

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(256), avg_hue_scores, label='Hue')
    plt.plot(range(256), avg_saturation_scores, label='Saturation')
    plt.plot(range(256), avg_value_scores, label='Value')
    plt.title('Average Score Evolution across Hue, Saturation, and Value')
    plt.xlabel('Threshold Value')
    plt.ylabel('Average Score')
    plt.legend()
    plt.grid(True)
    plt.show()

    return optimal_thresholds'''

def manual_segmentation_comparison(video_path, background_model_path, manual_mask_path, initial_ranges, initial_step=50, refinement_steps=[10, 5]):
    optimal_thresholds = None
    optimal_score = float('inf')

    # Ensure the initial_ranges dictionary includes a start and end for each channel, e.g., {'hue': (0, 255), ...}
    current_ranges = initial_ranges
    current_step = initial_step

    for refinement_step in refinement_steps + [1]:  # Ensure a final fine-grained search
        print(f"\nSearching with step size {current_step}")
        start_time = time.time()  # Start timing the search iteration

        # Adjust the search ranges based on the previous optimal thresholds
        search_ranges = {
            'hue': range(max(0, current_ranges['hue'][0]), min(256, current_ranges['hue'][1] + current_step), current_step),
            'saturation': range(max(0, current_ranges['saturation'][0]), min(256, current_ranges['saturation'][1] + current_step), current_step),
            'value': range(max(0, current_ranges['value'][0]), min(256, current_ranges['value'][1] + current_step), current_step),
        }

        # Nested loops for searching threshold values, wrapped with tqdm for progress tracking
        for h_thresh in tqdm(range(*search_ranges['hue']), desc="Hue Progress"):
            for s_thresh in tqdm(range(*search_ranges['saturation']), desc="Saturation Progress", leave=False):
                for v_thresh in tqdm(range(*search_ranges['value']), desc="Value Progress", leave=False):
                    segmented = background_subtraction(video_path, background_model_path, h_thresh, s_thresh, v_thresh, thresh_search=True)
                    xor_result = cv.bitwise_xor(segmented, cv.imread(manual_mask_path, 0))
                    score = cv.countNonZero(xor_result)

                    if score < optimal_score:
                        optimal_score = score
                        optimal_thresholds = (h_thresh, s_thresh, v_thresh)

        # Print the optimal thresholds found in this iteration
        print(f"Optimal thresholds after refinement with step {current_step}: Hue={optimal_thresholds[0]}, Saturation={optimal_thresholds[1]}, Value={optimal_thresholds[2]}")
        print(f"Iteration took {time.time() - start_time:.2f} seconds.")

        # Update search ranges for the next iteration
        current_ranges = {
            'hue': (max(0, optimal_thresholds[0] - current_step), optimal_thresholds[0] + current_step),
            'saturation': (max(0, optimal_thresholds[1] - current_step), optimal_thresholds[1] + current_step),
            'value': (max(0, optimal_thresholds[2] - current_step), optimal_thresholds[2] + current_step),
        }

        # Decrease step size for the next iteration
        current_step = refinement_step

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

def manual_segmentation_comparison(video_path, background_model_path, manual_mask_path, initial_ranges, initial_step=50, refinement_steps=[10, 5]):
    optimal_thresholds = None
    optimal_score = float('inf')

    # Ensure the initial_ranges dictionary includes a start and end for each channel, e.g., {'hue': (0, 255), ...}
    current_ranges = initial_ranges
    current_step = initial_step

    for refinement_step in refinement_steps + [1]:  # Ensure a final fine-grained search
        print(f"\nSearching with step size {current_step}")
        start_time = time.time()  # Start timing the search iteration

        # Adjust the search ranges based on the previous optimal thresholds
        search_ranges = {
            'hue': range(max(0, current_ranges['hue'][0]), min(256, current_ranges['hue'][1] + current_step), current_step),
            'saturation': range(max(0, current_ranges['saturation'][0]), min(256, current_ranges['saturation'][1] + current_step), current_step),
            'value': range(max(0, current_ranges['value'][0]), min(256, current_ranges['value'][1] + current_step), current_step),
        }

        # Nested loops for searching threshold values, wrapped with tqdm for progress tracking
        for h_thresh in tqdm(range(*search_ranges['hue']), desc="Hue Progress"):
            for s_thresh in tqdm(range(*search_ranges['saturation']), desc="Saturation Progress", leave=False):
                for v_thresh in tqdm(range(*search_ranges['value']), desc="Value Progress", leave=False):
                    segmented = background_subtraction(video_path, background_model_path, h_thresh, s_thresh, v_thresh, thresh_search=True)
                    xor_result = cv.bitwise_xor(segmented, cv.imread(manual_mask_path, 0))
                    score = cv.countNonZero(xor_result)

                    if score < optimal_score:
                        optimal_score = score
                        optimal_thresholds = (h_thresh, s_thresh, v_thresh)

        # Print the optimal thresholds found in this iteration
        print(f"Optimal thresholds after refinement with step {current_step}: Hue={optimal_thresholds[0]}, Saturation={optimal_thresholds[1]}, Value={optimal_thresholds[2]}")
        print(f"Iteration took {time.time() - start_time:.2f} seconds.")

        # Update search ranges for the next iteration
        current_ranges = {
            'hue': (max(0, optimal_thresholds[0] - current_step), optimal_thresholds[0] + current_step),
            'saturation': (max(0, optimal_thresholds[1] - current_step), optimal_thresholds[1] + current_step),
            'value': (max(0, optimal_thresholds[2] - current_step), optimal_thresholds[2] + current_step),
        }

        # Decrease step size for the next iteration
        current_step = refinement_step

    return optimal_thresholds
    
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
    
    # background subtraction
    #video_frame_path = cc.get_images_from_video(camera_number, f'data/cam{camera_number}/video.avi', test_image=True)
    #video_frame = cv.imread(video_frame_path)
    video_path = f'data/cam{camera_number}/video.avi'
    background_model_path = f'data/cam{camera_number}/background_model.jpg' 
    # Deprecated (maybe)
    # background_subtraction_model = background_subtraction(video_path, background_model_path)
    
    # manual subtraction
    frame_path = f'data/cam{camera_number}/actual_frame.jpg'
    frame = cv.imread(frame_path)  
    manual_mask_path = f'data/cam{camera_number}/manual_mask.jpg'   
    '''threshold_ranges = {
        'hue': range(0, 256, 5),
        'saturation': range(0, 256, 5),
        'value': range(0, 256, 5)
    }'''
    initial_ranges = {
        'hue': range(0, 256, 50),
        'saturation': range(0, 256, 50),
        'value': range(0, 256, 50)
    }
    #optimal_thresholds = manual_segmentation_comparison(video_path, background_model_path, manual_mask_path, threshold_ranges)
    optimal_thresholds = manual_segmentation_comparison(video_path, background_model_path, manual_mask_path, initial_ranges, initial_step=50, refinement_steps=[10, 5])
    print(f'Optimal thresholds: Hue={optimal_thresholds[0]}, Saturation={optimal_thresholds[1]}, Value={optimal_thresholds[2]}')
