import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
import math

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
            cv.namedWindow('img', cv.WINDOW_NORMAL)
            cv.imshow('img', img)
            print("\tCorner found at: ", x, y)
        if click == 4:
            # Check conditions
            p1, p2, p3, p4 = manual_coordinates
            
            if not (p1[0] < p2[0] and p2[1] < p3[1] and p3[0] > p4[0] and p4[1] > p1[1]):
                print("The points are not in the correct order, restart from the first")
                click = 0
                manual_coordinates = np.zeros((4, 2), dtype=np.float32)
            else:
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
                cv.namedWindow('img', cv.WINDOW_NORMAL)
                cv.imshow('img', img)

def manual_corners_selection(gray, img):
    global click
    global manual_coordinates
    global height
    global width
    global corners

    print("\tSelect 4 corners in the image in the order: top-left, top-right, bottom-right, bottom-left.")
    print("\tThe corners should be selected in a clockwise order and the first selected side should be long ", width, " squares.")
    input("\tPress Enter to continue...")
    print("\tClick on the image to select the corners...")
    correct_corners = False

    # Setting mouse handler for the image 
    cv.namedWindow('img', cv.WINDOW_NORMAL)
    cv.imshow('img', img)
    cv.setMouseCallback('img', click_event) # Limit to 4 clicks
    cv.waitKey(0) # Press any key to continue after 4 clicks
    cv.destroyAllWindows()

    # Perspective transformation
    target_height, target_width = img.shape[:2]  # You can set a specific size
    new_corners = np.float32([[0, 0], [target_width, 0], [target_width, target_height], [0, target_height]])
    # Calculate perspective transform matrix
    matrix = cv.getPerspectiveTransform(manual_coordinates, new_corners)

    warped_corners = cv.perspectiveTransform(manual_coordinates.reshape(-1, 1, 2), matrix)
    click = 0
    manual_coordinates = np.zeros((4, 2), dtype=np.float32)                             


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
    
    # Define the grid coordinates in the warped image
    grid_points = np.zeros(((height+1) * (width+1), 2), dtype=np.float32)

    # Generate grid coordinates
    index = 0
    for j in range(width+1):
        for i in range(height, -1, -1):
            x = j * width_warp/width
            y = i * height_warp/height
            grid_points[index] = (x, y)
            index += 1

    matrix_inv = cv.invert(matrix)[1]
    corners = cv.perspectiveTransform(grid_points.reshape(-1, 1, 2), matrix_inv)
    
    objpoints.append(objp)
    corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners2)

    cv.namedWindow('img', cv.WINDOW_NORMAL)
    cv.drawChessboardCorners(img, (9,6), corners2, True)
    cv.imshow('img', img)
    cv.waitKey(1000)
    cv.destroyAllWindows()

    add_corners_show_image(gray, img)

    # Reset click count and manual_coordinates for the next image
    click = 0
    manual_coordinates = np.zeros((4, 2), dtype=np.float32)

def add_corners_show_image(gray, img):
    global objp
    global objpoints
    global imgpoints
    global criteria
    global corners

    objpoints.append(objp)
    corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners2)
    # Draw and display the corners
    cv.namedWindow('img', cv.WINDOW_NORMAL)
    cv.drawChessboardCorners(img, (9,6), corners2, ret)
    cv.imshow('img', img)
    cv.waitKey(1000)
    cv.destroyAllWindows()


def process_frame(img, objp, criteria, mtx, dist, axis, cube_points):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners_test = cv.findChessboardCorners(gray, (height+1,width+1), None, cv.CALIB_CB_FAST_CHECK)
    if ret:
        corners2 = cv.cornerSubPix(gray, corners_test, (11,11), (-1,-1), criteria)
        # Find the rotation and translation vectors.
        ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
        imgpts_cube, jac = cv.projectPoints(cube_points, rvecs, tvecs, mtx, dist)
        img = draw(img, corners2, imgpts)
        img = draw_cube(img, imgpts_cube, color=(255, 255, 0), thickness=3)
    return img

def cross_validate_calibration(objpoints, imgpoints, image_size, k=5):
    # Ensure there are enough data points for k-fold
    if len(objpoints) < k or len(imgpoints) < k:
        raise ValueError("\tNot enough data points for k-fold cross-validation")

    # Calculate the size of each fold
    fold_size = len(objpoints) // k
    reprojection_errors = []

    for i in range(k):
        # Split the dataset into training and validation sets
        validation_objpoints = objpoints[i * fold_size: (i + 1) * fold_size]
        validation_imgpoints = imgpoints[i * fold_size: (i + 1) * fold_size]
        
        training_objpoints = objpoints[:i * fold_size] + objpoints[(i + 1) * fold_size:]
        training_imgpoints = imgpoints[:i * fold_size] + imgpoints[(i + 1) * fold_size:]
        
        # Calibrate the camera using the training set
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(training_objpoints, training_imgpoints, image_size, None, None)
        
        # Compute the reprojection error using the validation set
        if ret:
            mean_error = 0
            for j in range(len(validation_objpoints)):
                imgpoints2, _ = cv.projectPoints(validation_objpoints[j], rvecs[j], tvecs[j], mtx, dist)
                error = cv.norm(validation_imgpoints[j], imgpoints2, cv.NORM_L2) / len(imgpoints2)
                mean_error += error
            mean_error /= len(validation_objpoints)
            reprojection_errors.append(mean_error)

    # Calculate the mean and standard deviation of the reprojection errors
    mean_reprojection_error = np.mean(reprojection_errors)
    std_reprojection_error = np.std(reprojection_errors)
    return mean_reprojection_error, std_reprojection_error

def reject_low_quality_images(ret, mtx, dist, rvecs, tvecs):
    global objpoints
    global imgpoints
    global error_threshold

    print("Rejecting low-quality images...")
    # Calculate re-projection errors
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error

    mean_error /= len(objpoints)
    print(f"\tTotal error: {mean_error}")

    # Iterate to reject low-quality images
    while mean_error > error_threshold and len(objpoints) > 0:
        max_error_idx = -1
        max_error_value = 0

        # Find the image with the highest error
        for i in range(len(objpoints)):
            imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
            if error > max_error_value:
                max_error_value = error
                max_error_idx = i
        
        # If the worst image is above the threshold, remove it
        if max_error_value > error_threshold:
            print(f"\tRejecting image with error {max_error_value}")
            objpoints.pop(max_error_idx)
            imgpoints.pop(max_error_idx)
            # Re-calibrate without the worst image
            ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image_size, None, None)

            # Recalculate mean error
            mean_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
                mean_error += error

            mean_error /= len(objpoints)
            print(f"\n\tTotal error after rejection: {mean_error}")
        else:
            break  # No more images to reject

    print("\n")
    return ret, mtx, dist, rvecs, tvecs

def plot_camera_positions(tvecs):
    print("Plotting camera positions...")
    # Plot the origin of the chessboard
    camera_positions = np.array([tvec.flatten() for tvec in tvecs])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_title('Camera Positions Relative to Chessboard', fontsize=14)
    ax.scatter(0, 0, 0, c='red', marker='o', s=50, label='Chessboard Origin')
    ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], c='blue', marker='^', s=50, label='Camera Positions')

    # Connect the camera positions to the origin to show the relative positions
    for pos in camera_positions:
        ax.plot([0, pos[0]], [0, pos[1]], [0, pos[2]], 'gray', linestyle='--')
    N = square_size * width
    X, Y = np.meshgrid(np.linspace(-N/2, N/2, 10), np.linspace(-N/2, N/2, 10))
    Z = np.zeros_like(X)

    # Plot the grid plane (assuming the chessboard is at z=0)
    ax.plot_surface(X, Y, Z, alpha=0.2, color='green')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.legend()
    print("\tClose the plot window to continue...")
    plt.show()
    print("\n")

def draw(img, corners2, imgpts):
    def tupleOfInts(arr):
        return tuple(int(x) for x in arr)
    
    corner = tupleOfInts(corners2[0].ravel())

    img = cv.line(img, corner, tupleOfInts(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tupleOfInts(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tupleOfInts(imgpts[2].ravel()), (0,0,255), 5)
    return img

def draw_cube(img, imgpts_cube, color=(0, 255, 0), thickness=3):
    imgpts_cube = imgpts_cube.reshape(-1, 2)
    imgpts_cube = np.int32(imgpts_cube)

    # Draw bottom face
    for i in range(4):
        next_index = (i + 1) % 4
        img = cv.line(img, tuple(imgpts_cube[i]), tuple(imgpts_cube[next_index]), color, thickness)
    
    # Draw top face
    for i in range(4, 8):
        next_index = 4 + ((i + 1) % 4)
        img = cv.line(img, tuple(imgpts_cube[i]), tuple(imgpts_cube[next_index]), color, thickness)

    # Draw vertical lines (pillars)
    for i in range(4):
        img = cv.line(img, tuple(imgpts_cube[i]), tuple(imgpts_cube[i + 4]), color, thickness)

    return img

# Settings

# Set to True to use the webcam for testing, or False to use the test static image
use_webcam = False
# Set the error threshold for rejecting low-quality images
error_threshold = 0.5
# Define the width and height of the internal chessboard (in squares)
width = 5
height = 8
# Define the size of the squares of the chessboard in meters
square_size = 0.022


# Number of clicks for the manual selection of corners
click = 0
# Array to store the manual coordinates
manual_coordinates = np.zeros((4, 2), dtype=np.float32)
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points
objp = np.zeros(((width+1)*(height+1), 3), np.float32)
objp[:,:2] = np.mgrid[0:(height+1),0:(width+1)].T.reshape(-1,2) * square_size # Adjusted to the grid real dimensions

image_size = None # TODO: reposition and rename

# Initialize lists to store the calibration results for the runs
ret_list = []
mtx_list = []
dist_list = []
rvecs_list = []
tvecs_list = []

runs = ['Chessboard*.jpg', '*_selected.jpg', '*_more_selected.jpg'] # Run1: all 25 images, Run2: 10 images corners auto, Run3: 5 images corners auto
for run_n, run in enumerate(runs):
    images = glob.glob(run)
    print("\n\n")
    print("-" * 50) # Print a separator
    print("Run ", run_n+1, " of ", len(runs), " runs.")

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    for fname in images:
        print("Processing image: ", fname)
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (height+1,width+1), None, cv.CALIB_CB_FAST_CHECK)
        # If found corners, add object points, image points (after refining them) otherwise manual select them
        if ret == True:
            print("\tAuto corners: ", fname)
            add_corners_show_image(gray, img)
        else:
            print("\tManual corners: ", fname)
            manual_corners_selection(gray, img)
        
        if image_size is None:
           image_size = gray.shape[::-1]

        print("\n")
        
    if image_size is not None:

        # Cross-validation to check the quality of the calibration
        if len(objpoints) >= 5:  # Ensure there are enough points for cross-validation
            print("Cross-validating the calibration...")
            mean_error, std_error = cross_validate_calibration(objpoints, imgpoints, image_size, k=5)
            print(f"\tMean reprojection error from cross-validation: {mean_error}")
            print(f"\tStandard deviation of reprojection error from cross-validation: {std_error}")
            print("\n")
        
        # Calibrate the camera
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image_size, None, None)

        # Reject low-quality images and recalibrate
        ret, mtx, dist, rvecs, tvecs = reject_low_quality_images(ret, mtx, dist, rvecs, tvecs)

        objp_test = np.zeros(((width+1)*(height+1),3), np.float32)
        objp_test[:,:2] = np.mgrid[0:(height+1),0:(width+1)].T.reshape(-1,2)
        axis = np.float32([[4,0,0], [0,4,0], [0,0,-4]]).reshape(-1,3)
        cube_size = 2
        cube_points = np.float32([
        [0, 0, 0], [0, cube_size, 0], [cube_size, cube_size, 0], [cube_size, 0, 0],
        [0, 0, -cube_size], [0, cube_size, -cube_size], [cube_size, cube_size, -cube_size], [cube_size, 0, -cube_size]
        ])

        # Main processing loop
        if use_webcam:
            print("Using the webcam...")
            print("\tPress q to continue")
            # When using the webcam
            cap = cv.VideoCapture(0)
            if not cap.isOpened():
                print("\tCannot open camera")
                exit()
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("\tCan't receive frame (stream end?). Exiting ...")
                    break
                # Process the captured frame
                frame = process_frame(frame, objp_test, criteria, mtx, dist, axis, cube_points)
                cv.imshow('Webcam', frame)
                if cv.waitKey(1) == ord('q'):
                    break
            cap.release()
            cv.destroyAllWindows()
            print("\n")
        else:
            # When using static images
            for fname in glob.glob('Test_image.jpg'):
                print("Using the test static image...", fname)
                print("\tPress 's' to save the image or any other key to continue")
                img = cv.imread(fname)
                img = process_frame(img, objp_test, criteria, mtx, dist, axis, cube_points)
                cv.namedWindow('img', cv.WINDOW_NORMAL)
                cv.imshow('img', img)
                k = cv.waitKey(0) & 0xFF
                if k == ord('s'):
                    cv.imwrite('Result_image_run_'+run_n+'.png', img)
                cv.destroyAllWindows()
            print("\n")
        
        plot_camera_positions(tvecs)

        ret_list.append(ret)
        mtx_list.append(mtx)
        dist_list.append(dist)
        rvecs_list.append(rvecs)
        tvecs_list.append(tvecs)

        print("\n")
        print("Thanks for using the camera calibration tool!")

cv.destroyAllWindows()




