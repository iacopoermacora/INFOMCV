import camera_calibration as cc
import settings as settings
import cv2 as cv
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
    
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

    ret, frame = cap.read() # TODO: I removed the loop, this way it is done only on the first image, I am not sure if this is the correct way to do it or if we need to use thw whole video in the future

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
    min_blob_area_1 = 1000  # Adjust this threshold as needed
    blobs_1 = [cnt for cnt in contours_1 if cv.contourArea(cnt) > min_blob_area_1]

    # Draw the detected blobs
    blob_mask = np.zeros_like(dilation_mask)
    cv.drawContours(blob_mask, blobs_1, -1, (255), thickness=cv.FILLED)

    dilation_mask_bitw = cv.bitwise_and(threshold_mask, blob_mask)
    '''# Dilation to fill in gaps
    kernel_2 = np.ones((2, 2), np.uint8)
    dilation_mask_2 = cv.dilate(dilation_mask_bitw, kernel_2, iterations=1)

    
    # BLOB DETECTION AND REMOVAL
    # Find contours
    contours_2, _ = cv.findContours(dilation_mask_2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area to identify blobs
    min_blob_area_2 = 20  # Adjust this threshold as needed
    blobs_2 = [cnt for cnt in contours_2 if cv.contourArea(cnt) > min_blob_area_2]

    # Draw the detected blobs
    blob_mask_2 = np.zeros_like(dilation_mask_2)
    cv.drawContours(blob_mask_2, blobs_2, -1, (255), thickness=cv.FILLED)
    
    combined_mask = blob_mask_2'''
    combined_mask = dilation_mask_bitw # TODO: Remove when reimplementing the part before


    '''if thresh_search == False:
        #cv.imshow('threshold Mask', threshold_mask)
        cv.imshow('dilation Mask', dilation_mask)
        cv.imshow('Blob Mask', blob_mask)
        cv.imshow('bitw Mask', dilation_mask_bitw)
        cv.imshow('dilation Mask 2', dilation_mask_2)
        cv.imshow('Foreground Mask', combined_mask)
        cv.waitKey(0) 

    cap.release()
    cv.destroyAllWindows()
    cv.waitKey(1)'''

    return combined_mask

def manual_segmentation_comparison(video_path, background_model_path, manual_mask_path, camera_number, steps=[50, 10, 5, 1]):
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

    # Dilation to fill in gaps TODO: Remove from here
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

    segmented = blob_mask_2 # TODO: Remove until here

    cv.imshow('Foreground Mask', segmented)
    cv.waitKey(0) 
    cv.destroyAllWindows()
    cv.waitKey(1)

    cv.imwrite(f'data/cam{camera_number}/foreground_mask.jpg', segmented)

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
    

# Call the function to get the camera intrinsics and extrinsics for each camera
for camera_number in range(1, settings.num_cameras+1):
    cc.get_camera_intrinsics_and_extrinsics(camera_number)
    # background model
    background_video_path = f'data/cam{camera_number}/background.avi'
    background_model = create_background_model_gmm(background_video_path)
    
    # display of the background model
    # if background_model is not None:
    #     cv.imshow('Background Model', background_model)
    #     cv.waitKey(0) 
    #     cv.destroyAllWindows()
    # else:
    #     print("Failed to create background model")
    
    # manual subtraction
    manual_mask_path = f'data/cam{camera_number}/manual_mask.jpg'
    video_path = f'data/cam{camera_number}/video.avi'
    background_model_path = f'data/cam{camera_number}/background_model.jpg'
    optimal_thresholds = manual_segmentation_comparison(video_path, background_model_path, manual_mask_path, camera_number, steps=[50, 10, 5, 1])

