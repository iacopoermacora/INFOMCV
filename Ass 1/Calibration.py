import numpy as np
import cv2 as cv
import glob

manual_coordinates = np.zeros((4, 2), dtype=np.float32)
click = 0

def click_event(event, x, y, flags, params): 
    global click
    global manual_coordinates

    # Checking for left mouse clicks 
    if event == cv.EVENT_LBUTTONDOWN: 
        if click < 4:  # Ensure only 4 points are selected
            manual_coordinates[click] = (x, y)
            click += 1
            print("Corner found at: ", x, y)

# Define the width and height of the chessboard (in squares)
width = 5
height = 8

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((width*height,3), np.float32)
objp[:,:2] = np.mgrid[0:height,0:width].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('*.JPG')
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (height,width), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (9,6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(0)
    else:
        print("Manual corners:")
        # Setting mouse handler for the image 
        cv.namedWindow('img', cv.WINDOW_NORMAL)
        cv.imshow('img', img)
        cv.setMouseCallback('img', click_event) # Limit to 4 clicks
        cv.waitKey(0) # Press any key to continue after 4 clicks
        cv.destroyAllWindows()

        if click < 4:
            print("Insufficient points selected.")
            continue
        h, w = img.shape[:2]
        # Perspective transformation
        target_height, target_width = img.shape[:2]  # You can set a specific size
        new_corners = np.float32([[0, 0], [target_width, 0], [target_width, target_height], [0, target_height]])
        # Calculate perspective transform matrix
        matrix = cv.getPerspectiveTransform(manual_coordinates, new_corners)
        # Apply perspective transform
        warped_image = cv.warpPerspective(img, matrix, (target_width, target_height))

        warped_corners = cv.perspectiveTransform(manual_coordinates.reshape(-1, 1, 2), matrix)

        # Initialize variables to store the width and height of the warped image
        width_warp = 0
        height_warp = 0

        # Iterate through warped_corners to find width and height
        for point in warped_corners:
            x, y = point[0]
            if y > height_warp:
                height_warp = y
            if x > width_warp:
                width_warp = x
        
        # Define the grid coordinates
        grid_points = np.zeros(((height+1) * (width+1), 2), dtype=np.float32)

        # Generate grid coordinates
        index = 0
        for i in range(height+1):
            for j in range(width+1):
                x = j * width_warp/width
                y = i * height_warp/height
                grid_points[index] = (x, y)  # Adjust 100 according to your grid spacing
                index += 1

        Minv = cv.invert(matrix)[1]
        original_points = cv.perspectiveTransform(grid_points.reshape(-1, 1, 2), Minv)
        
        cv.namedWindow('img', cv.WINDOW_NORMAL)
        cv.drawChessboardCorners(img, (9,6), original_points, True)
        cv.imshow('img', img)
        cv.waitKey(0)

        # Reset click count and manual_coordinates for the next image
        click = 0
        manual_coordinates = np.zeros((4, 2), dtype=np.float32)

cv.destroyAllWindows()