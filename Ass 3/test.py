import numpy as np
import cv2 as cv

video_path_color = f'data/cam1/video.avi'
cap_color = cv.VideoCapture(video_path_color)
cap_color.set(cv.CAP_PROP_POS_FRAMES, 0)
_, frame_color = cap_color.read()
# Convert frame to HSV
frame_hsv = cv.cvtColor(frame_color, cv.COLOR_BGR2HSV)
print(frame_hsv[30, 45])
print(frame_color[30, 45])
