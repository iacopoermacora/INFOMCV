import numpy as np
import cv2 as cv
import glob

def click_event(event, x, y, flags, params): 
    global click
    global manual_coordinates
    global img

    # Checking for left mouse clicks 
    if event == cv.EVENT_LBUTTONDOWN: 
        if click < 4:  # Ensure only 4 points are selected
            manual_coordinates[click] = (x, y)
            click += 1
            cv.circle(img, (x, y), 10, (0, 0, 255), -1)
            cv.imshow('img', img)
            print("Corner found at: ", x, y)
        if click == 4:
            # font 
            font = cv.FONT_HERSHEY_SIMPLEX 
            # org 
            org = (100, 500) 
            # fontScale 
            fontScale = 3
            # Blue color in BGR 
            color = (255, 0, 0) 
            # Line thickness of 2 px 
            thickness = 10
            cv.putText(img, 'Press any key to find all chessboard corners', org, font, fontScale, color, thickness, cv.LINE_AA)
            cv.imshow('img', img)

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

# Define the width and height of the internal chessboard (in squares)
width = 5
height = 8
click = 0
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
image_size = None

ret_list = []
mtx_list = []
dist_list = []
rvecs_list = []
tvecs_list = []

images_names = ['*_selected.jpg', '*_more_selected.jpg'] # '*.jpg'
for images_name in images_names:
    images = glob.glob(images_name)
    print(images_name)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    square_size = 0.022
    objp = np.zeros(((width+1)*(height+1),3), np.float32)
    objp[:,:2] = np.mgrid[0:(height+1),0:(width+1)].T.reshape(-1,2) * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    manual_coordinates = np.zeros((4, 2), dtype=np.float32)

    for fname in images:
        print(fname)
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (height+1,width+1), None, cv.CALIB_CB_FAST_CHECK)
        # ret = False
        # If found, add object points, image points (after refining them)
        if ret == True:
            print("Auto corners: ", fname)
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv.namedWindow('img', cv.WINDOW_NORMAL)
            cv.drawChessboardCorners(img, (9,6), corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(1000)
            cv.destroyAllWindows()
        else:
            print("Manual corners: ", fname)
            print("Select 4 corners in the image in the order: top-left, top-right, bottom-right, bottom-left")
            input("Press Enter to continue...")
            # Setting mouse handler for the image 
            cv.namedWindow('img', cv.WINDOW_NORMAL)
            cv.imshow('img', img)
            cv.setMouseCallback('img', click_event) # Limit to 4 clicks
            cv.waitKey(0) # Press any key to continue after 4 clicks
            cv.destroyAllWindows()

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
            for j in range(width+1):
                for i in range(height, -1, -1):
                    x = j * width_warp/width
                    y = i * height_warp/height
                    grid_points[index] = (x, y)  # Adjust 100 according to your grid spacing
                    index += 1

            Minv = cv.invert(matrix)[1]
            original_points = cv.perspectiveTransform(grid_points.reshape(-1, 1, 2), Minv)
            
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, original_points, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            cv.namedWindow('img', cv.WINDOW_NORMAL)
            cv.drawChessboardCorners(img, (9,6), original_points, True)
            cv.imshow('img', img)
            cv.waitKey(1000)
            cv.destroyAllWindows()

            # Reset click count and manual_coordinates for the next image
            click = 0
            manual_coordinates = np.zeros((4, 2), dtype=np.float32)
        
        if image_size is None:
           image_size = gray.shape[::-1] 
        
    if image_size is not None:
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros(((width+1)*(height+1),3), np.float32)
        objp[:,:2] = np.mgrid[0:(height+1),0:(width+1)].T.reshape(-1,2) * square_size
        axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

        for fname in glob.glob('Chessboard_9_more_selected.jpg'):
            img = cv.imread(fname)
            gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(gray, (height+1,width+1), None, cv.CALIB_CB_FAST_CHECK)
            if ret == True:
                corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                # Find the rotation and translation vectors.
                ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
                # project 3D points to image plane
                imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
                img = draw(img,corners2,imgpts)
                print("GETS HERE")
                cv.imshow('img',img)
                k = cv.waitKey(0) & 0xFF
                if k == ord('s'):
                    cv.imwrite(fname[:6]+'.png', img)
            cv.destroyAllWindows()

        ret_list.append(ret)
        mtx_list.append(mtx)
        dist_list.append(dist)
        rvecs_list.append(rvecs)
        tvecs_list.append(tvecs)

cv.destroyAllWindows()