import numpy as np
import cv2 as cv
import glob

manual_coordinates = np.zeros((4, 2), dtype=np.float32)
click = 0

def click_event(event, x, y, flags, params): 
    global click
    global manual_coordinates

    # checking for left mouse clicks 
    if event == cv.EVENT_LBUTTONDOWN: 
        manual_coordinates[click] = (x, y)
        click += 1
        print("Corner found at: ", x, y)

# Define the width and height of the chessboard (in squares)
width = 6
height = 9

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((width*height,3), np.float32)
objp[:,:2] = np.mgrid[0:height,0:width].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('*.png')
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (height,width), None)
    print(corners.shape)
    print("Manual corners:")
    # setting mouse handler for the image 
    # and calling the click_event() function 
    cv.imshow('img', img)
    cv.setMouseCallback('img', click_event) # Limit to 4 clicks
    cv.waitKey(0) # Press any key to continue
    print(manual_coordinates)

    # Calculate the number of tiles
    num_tiles = width * height

    # Define the new positions of the corners after transformation
    new_corners = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])

    # Calculate perspective transform matrix
    matrix = cv.getPerspectiveTransform(manual_coordinates, new_corners)

    # Apply perspective transform
    warped_image = cv.warpPerspective(img, matrix, (width, height))

    # Calculate the coordinates of the intersections
    intersections = []
    for i in range(height):
        for j in range(width):
            intersections.append((j, i))
    
    print(intersections)

    # Display the result
    cv.imshow('Warped Image', warped_image)
    cv.waitKey(0)


    # # If found, add object points, image points (after refining them)
    # if ret == True:
    #     objpoints.append(objp)
    #     corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    #     imgpoints.append(corners2)
    #     # Draw and display the corners
    #     cv.drawChessboardCorners(img, (9,6), corners2, ret)
    #     cv.imshow('img', img)
    #     cv.waitKey(0)
cv.destroyAllWindows()