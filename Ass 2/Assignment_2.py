import camera_calibration as cc
import settings as settings
import cv2 as cv
import numpy as np
import os

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
    backSub = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    while True:
        ret, frame = cap.read()
        if ret:
            cv.imshow('Captured Frame', frame)
        if not ret:
            break
        # Apply the background subtractor to each frame
        fgMask = backSub.apply(frame)
    # Retrieve the final background model after processing all frames
    background_model = backSub.getBackgroundImage()
    
    cap.release()
    return background_model

def background_subtraction(video_path, background_model_path):
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
        _, thresh_h = cv.threshold(diff[:,:,0], 50, 255, cv.THRESH_BINARY)
        _, thresh_s = cv.threshold(diff[:,:,1], 50, 255, cv.THRESH_BINARY)
        _, thresh_v = cv.threshold(diff[:,:,2], 50, 255, cv.THRESH_BINARY)


        # Combine the thresholds (example using logical AND)
        combined_mask = cv.bitwise_and(thresh_h, cv.bitwise_and(thresh_s, thresh_v))

    cv.imshow('Foreground Mask', combined_mask)
    cv.waitKey(0) 
    cv.destroyAllWindows()

    cap.release()
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

cam_calibration_data = []
# Call the function to get the camera intrinsics and extrinsics for each camera
for camera_number in range(1, settings.num_cameras+1):
    cc.get_camera_intrinsics_and_extrinsics(camera_number)
    # background_video_path = f'data/cam{camera_number}/background.avi'
    # background_model = create_background_model_gmm(background_video_path)
    
    # # display of the background model
    # if background_model is not None:
    #     cv.imshow('Background Model', background_model)
    #     cv.waitKey(0) 
    #     cv.destroyAllWindows()
    # else:
    #     print("Failed to create background model")
    
    # video_path = f'data/cam{camera_number}/video.avi'
    # background_model_path = f'data/cam{camera_number}/background_model.jpg'  # Update this path
    # background_subtraction_model = background_subtraction(video_path, background_model_path)

    # Part 3: Voxel reconstruction
    cam_calibration_data.append(read_camera_parameters(camera_number))

# Construct lookup table
voxel_volume = np.zeros((100, 100, 100))  # Example voxel volume shape
lookup_table = construct_lookup_table(voxel_volume, cam_calibration_data, num_views=4)
    

