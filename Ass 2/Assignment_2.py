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
    combined_mask = cv.bitwise_and(thresh_v, cv.bitwise_and(thresh_h, thresh_s))

    # # Dilation to fill in gaps
    # kernel = np.ones((3, 3), np.uint8)
    # combined_mask = cv.dilate(combined_mask, kernel, iterations=1)

    # # BLOB DETECTION AND REMOVAL
    # # Find contours
    # contours, _ = cv.findContours(combined_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # # Filter contours based on area to identify blobs
    # min_blob_area = 50  # Adjust this threshold as needed
    # blobs = [cnt for cnt in contours if cv.contourArea(cnt) > min_blob_area]

    # # Draw the detected blobs
    # blob_mask = np.zeros_like(combined_mask)
    # cv.drawContours(blob_mask, blobs, -1, (255), thickness=cv.FILLED)

    # combined_mask = blob_mask
    
    # if thresh_search == True:
    #     break
    # else:
    #     cv.imshow('Foreground Mask before', combined_mask)
    #     cv.waitKey(0)
    #     # Perform graph cuts using GrabCut algorithm
    #     mask = np.zeros(frame.shape[:2], np.uint8)
    #     converted_mask = np.zeros_like(combined_mask, dtype=np.uint8)
    #     # Set background regions to GC_BGD
    #     converted_mask[combined_mask == 0] = cv.GC_FGD
    #     # Set foreground regions to GC_FGD
    #     converted_mask[combined_mask == 255] = cv.GC_BGD
    #     bgdModel = np.zeros((1,65),np.float64)
    #     fgdModel = np.zeros((1,65),np.float64)
    #     rect = (0, 0, frame.shape[1]-1, frame.shape[0]-1)  # Entire frame region
    #     mask, _, _ = cv.grabCut(frame, converted_mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_MASK)
    #     # Modify the mask to get the foreground
    #     combined_mask = np.where((mask==2)|(mask==0), 255, 0).astype('uint8')
    #     break

    if thresh_search == False:
        cv.imshow('Foreground Mask', combined_mask)
        cv.waitKey(0) 

    cap.release()
    cv.destroyAllWindows()
    cv.waitKey(1)

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
    

'''# Call the function to get the camera intrinsics and extrinsics for each camera
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
    optimal_thresholds = manual_segmentation_comparison(video_path, background_model_path, manual_mask_path, camera_number, steps=[50, 10, 5, 1])'''

