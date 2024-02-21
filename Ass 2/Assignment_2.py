import camera_calibration as cc
import settings as settings
import cv2 as cv
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

# Call the function to get the camera intrinsics and extrinsics for each camera
for camera_number in range(1, settings.num_cameras+1):
    background_video_path = f'data/cam{camera_number}/background.avi'
    background_model = create_background_model_gmm(background_video_path)
    
    # display of the background model
    if background_model is not None:
        cv.imshow('Background Model', background_model)
        cv.waitKey(0) 
        cv.destroyAllWindows()
    else:
        print("Failed to create background model")
    
    video_path = f'data/cam{camera_number}/video.avi'
    background_model_path = f'data/cam{camera_number}/background_model.jpg'  # Update this path
    background_subtraction_model = background_subtraction(video_path, background_model_path)
    cc.get_camera_intrinsics_and_extrinsics(camera_number)