import cv2
import numpy as np

# Read the video
cap = cv2.VideoCapture('video_data/clap/_Boom_Snap_Clap__challenge_clap_u_nm_np1_fr_med_0.avi')  # Replace 'input_video.mp4' with the path to your video file

# Get total number of frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Calculate middle frame index
middle_frame_index = total_frames // 2

# Set frame position to the middle frame
cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)

# Read the middle frame
ret, frame = cap.read()

# Convert frame to grayscale
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Parameters for Farneback optical flow
flow_params = dict(
    pyr_scale=0.5,
    levels=5,
    winsize=11,
    iterations=5,
    poly_n=5,
    poly_sigma=1.1,
    flags=0
)

# Calculate optical flow
next_frame_index = middle_frame_index + 1
cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame_index)
ret, next_frame = cap.read()
next_frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
flow = cv2.calcOpticalFlowFarneback(frame_gray, next_frame_gray, None, **flow_params)

# Visualize optical flow
h, w = frame_gray.shape
y, x = np.mgrid[0:h:10, 0:w:10].reshape(2, -1)
fx, fy = flow[y, x].T
lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
lines = np.int32(lines + 0.5)
vis = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
cv2.polylines(vis, lines, 0, (0, 255, 0))

# Display the visualization
cv2.imshow('Optical Flow', vis)
cv2.waitKey(0)

cv2.imshow('IMG1', frame_gray)
cv2.imshow('IMG2', next_frame_gray)
cv2.waitKey(0)

# Release resources
cap.release()
cv2.destroyAllWindows()