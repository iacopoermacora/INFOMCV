import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt

def click_event(event, x, y, flags, params): 
    '''
    click_event: Function to handle mouse clicks on the image

    :param event: The event type
    :param x: The x-coordinate of the mouse click
    :param y: The y-coordinate of the mouse click
    :param flags: The flags (not used in this assignment)
    :param params: The parameters (not used in this assignment)
    '''
    global click
    global manual_coordinates
    global img

    # Checking for left mouse clicks 
    if event == cv.EVENT_LBUTTONDOWN: 
        if click < 4:  # Ensure only 4 points are selected
            manual_coordinates[click] = (x, y)
            click += 1
            # Draw a small circle at the clicked point
            cv.circle(img, (x, y), 10, (0, 0, 255), -1)
            cv.namedWindow('img', cv.WINDOW_NORMAL)
            cv.imshow('img', img)
            print("\tCorner found at: ", x, y)
        if click == 4:
            # Check conditions
            p1, p2, p3, p4 = manual_coordinates
            
            # Check if the points are in the correct order
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
    '''
    manual_corners_selection: Function to manually select the corners of the chessboard

    :param gray: The grayscale image of the chessboard
    :param img: The original image of the chessboard

    :return: None (but save the four corners of the image in the global variable corners)
    '''
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

    # Display the image and wait for the user to select the corners
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
    # Warp perspective points
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

    # Calculate the inverse of the matrix
    matrix_inv = cv.invert(matrix)[1]
    # Calculate the corners in the original image
    corners = cv.perspectiveTransform(grid_points.reshape(-1, 1, 2), matrix_inv)

    # Draw the corners on the image
    add_corners_show_image(gray, img)

    # Reset click count and manual_coordinates for the next image
    click = 0
    manual_coordinates = np.zeros((4, 2), dtype=np.float32)

def add_corners_show_image(gray, img):
    '''
    add_corners_show_image: Function to add all the (inner) corners to the image and display it

    :param gray: The grayscale image of the chessboard
    :param img: The original image of the chessboard

    :return: None (but display the image with the corners)
    '''
    global objp
    global objpoints
    global imgpoints
    global criteria
    global corners
    global height
    global width

    # Save the object points
    objpoints.append(objp)
    # Improve the corner positions
    corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    # Save the image points
    imgpoints.append(corners2)

    # Draw and display the corners
    cv.namedWindow('img', cv.WINDOW_NORMAL)
    cv.drawChessboardCorners(img, (height+1,width+1), corners2, True)
    cv.imshow('img', img)
    cv.waitKey(1000)
    cv.destroyAllWindows()
    cv.waitKey(1)


def process_frame(img, objp, criteria, mtx, dist, axis, cube_points):
    '''
    process_frame: Function to process the frame and display the chessboard corners and the cube
    
    :param img: The image to process
    :param objp: The object points
    :param criteria: The termination criteria
    :param mtx: The camera matrix
    :param dist: The distortion coefficients
    :param axis: The axis points for the cube
    :param cube_points: The points for the cube

    :return: The image with the corners and the cube drawn on it
    '''
    # Convert the image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
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

def validate_calibration(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    '''
    validate_calibration: Function to validate the calibration by calculating the re-projection errors

    :param objpoints: The object points
    :param imgpoints: The image points
    :param rvecs: The rotation vectors
    :param tvecs: The translation vectors
    :param mtx: The camera matrix
    :param dist: The distortion coefficients

    :return: The mean re-projection error
    '''
    # Calculate re-projection errors
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error

    mean_error /= len(objpoints)
    return mean_error

def reject_low_quality_images(ret, mtx, dist, rvecs, tvecs):
    '''
    reject_low_quality_images: Function to reject low-quality images and recalibrate

    :param ret: The return value from the calibration
    :param mtx: The camera matrix
    :param dist: The distortion coefficients
    :param rvecs: The rotation vectors
    :param tvecs: The translation vectors

    :return: The return value from the calibration, the camera matrix, the distortion coefficients, the rotation vectors, and the translation vectors
    '''
    global objpoints
    global imgpoints
    global error_threshold

    print("Rejecting low-quality images...")
    mean_error = validate_calibration(objpoints, imgpoints, rvecs, tvecs, mtx, dist)
    print(f"\tMean reprojection error: {mean_error}")

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
            mean_error = validate_calibration(objpoints, imgpoints, rvecs, tvecs, mtx, dist)
            print(f"\tMean validation error after rejection: {mean_error}")
        else:
            print(f"\tNo more images to reject with error above {error_threshold}")
            break  # No more images to reject

    print("\n")
    return ret, mtx, dist, rvecs, tvecs

def plot_camera_positions(tvecs):
    '''
    plot_camera_positions: Function to plot the camera positions relative to the chessboard

    :param tvecs: The translation vectors

    :return: None (but display the plot)
    '''
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
    '''
    draw: Function to draw the axis on the image

    :param img: The image to draw on
    :param corners2: The corners of the chessboard
    :param imgpts: The image points

    :return: The image with the axis drawn on it
    '''
    def tupleOfInts(arr):
        return tuple(int(x) for x in arr)
    
    corner = tupleOfInts(corners2[0].ravel())

    img = cv.line(img, corner, tupleOfInts(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tupleOfInts(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tupleOfInts(imgpts[2].ravel()), (0,0,255), 5)
    return img

def draw_cube(img, imgpts_cube, color=(0, 255, 0), thickness=3):
    '''
    draw_cube: Function to draw the cube on the image

    :param img: The image to draw on
    :param imgpts_cube: The image points of the cube
    :param color: The color of the lines
    :param thickness: The thickness of the lines

    :return: The image with the cube drawn on it
    '''
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

print("\n")
print("Welcome to the camera calibration tool!")
print("This tool will help you calibrate your camera using a chessboard pattern.")
print("If you haven't already, please make sure to set the parameters in the script before running it and to upload the images in jpg format in the same folder as the script.")
print("Parameters to set: use_webcam, error_threshold, width, height, square_size, test_image")
print("\n")

# Parameters Settings

# Set to True to use the webcam for testing, or False to use the test static image
use_webcam = False
# Set the error threshold for rejecting low-quality images
error_threshold = 0.5
# Define the width and height of the internal chessboard (in squares)
width = 5
height = 8
# Define the size of the squares of the chessboard in meters
square_size = 0.022
# Define the test image to use for the static image test
test_image = "Test_image.jpg"


# Number of clicks for the manual selection of corners
click = 0
# Array to store the manual coordinates
manual_coordinates = np.zeros((4, 2), dtype=np.float32)
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points
objp = np.zeros(((width+1)*(height+1), 3), np.float32)
objp[:,:2] = np.mgrid[0:(height+1),0:(width+1)].T.reshape(-1,2) * square_size # Adjusted to the grid real dimensions
# Variable to store the image size
image_size = None

# Initialize lists to store the calibration results for the runs
ret_list = []
mtx_list = []
dist_list = []
rvecs_list = []
tvecs_list = []

# Define the image properties (based on the test image as all the images come from the same camera)
info_image = cv.imread(test_image)
# Image resolution (width, height)
height_image, width_image, _ = info_image.shape
image_resolution = (width_image, height_image)

runs = [] # Run1: all 25 images, Run2: 10 images corners auto, Run3: 5 images corners auto
images = glob.glob('*.jpg')
# Store the images for the 2nd and 3rd runs
images_2 = []
images_3 = []

for run in range(3):
    print("\n\n")
    print("-" * 50) # Print a separator
    print("Run ", run+1, " of 3 runs.")

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # Set the images to process based on the run
    if run == 1:
        images = images_2
    elif run == 2:
        images = images_3
    
    # Reset the counter for the auto-found corners
    counter_auto_found = 0

    for fname in images:

        print("Processing image: ", fname)
        # Read the image and convert to grayscale
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (height+1,width+1), None, cv.CALIB_CB_FAST_CHECK)
        # If found corners, add object points, image points (after refining them) otherwise manual select them
        if ret == True:
            # Create the image lists for the second and third run
            counter_auto_found += 1
            if counter_auto_found <= 10 and run == 0:
                images_2.append(fname)
                if counter_auto_found <= 5:
                    images_3.append(fname)
            
            print("\tAuto-detect corners: ", fname)
            add_corners_show_image(gray, img)
        else:
            print("\tUnfortunatelly, the corners were not found automatically. Please select them manually.")
            print("\tManually select corners: ", fname)
            manual_corners_selection(gray, img)
        
        if image_size is None:
           image_size = gray.shape[::-1]

        print("\n")
        
    if image_size is not None:
        
        # Calibrate the camera
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image_size, None, None)

        # Reject low-quality images and recalibrate
        ret, mtx, dist, rvecs, tvecs = reject_low_quality_images(ret, mtx, dist, rvecs, tvecs)

        print("Intrinsic camera matrix run ", (run+1), ": \n\t", mtx)

        # Assume mtx, rvecs, tvecs are obtained from cv.calibrateCamera()
        focal_length_x = mtx[0, 0]
        focal_length_y = mtx[1, 1]
        principal_point_x = mtx[0, 2]
        principal_point_y = mtx[1, 2]
        aspect_ratio = focal_length_x / focal_length_y

        print(f"\tFocal length in x (f_x): {focal_length_x}")
        print(f"\tFocal length in y (f_y): {focal_length_y}")
        print(f"\tPrincipal point x (c_x): {principal_point_x}")
        print(f"\tPrincipal point y (c_y): {principal_point_y}")
        print(f"\tAspect ratio: {aspect_ratio}")
        print(f"\tResolution of the images: {image_resolution}")
        print("\n")

        # Define the object points for the test image
        objp_test = np.zeros(((width+1)*(height+1),3), np.float32)
        objp_test[:,:2] = np.mgrid[0:(height+1),0:(width+1)].T.reshape(-1,2)
        # Define the axis
        axis = np.float32([[4,0,0], [0,4,0], [0,0,-4]]).reshape(-1,3)
        # Define the cube points
        cube_size = 2
        cube_points = np.float32([
        [0, 0, 0], [0, cube_size, 0], [cube_size, cube_size, 0], [cube_size, 0, 0],
        [0, 0, -cube_size], [0, cube_size, -cube_size], [cube_size, cube_size, -cube_size], [cube_size, 0, -cube_size]
        ])

        # Test the calibration using the webcam or the static test image
        if use_webcam:
            # When using the webcam
            print("Using the webcam...")
            print("\tPress q to continue")
            # Open the webcam
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
            print("Using the test static image...", test_image)
            print("\tPress 's' to save the image or any other key to continue")
            # Read and process the test image
            img = cv.imread(test_image)
            img = process_frame(img, objp_test, criteria, mtx, dist, axis, cube_points)
            cv.namedWindow('img', cv.WINDOW_NORMAL)
            cv.imshow('img', img)
            # Save the image if 's' is pressed
            k = cv.waitKey(0) & 0xFF
            if k == ord('s'):
                cv.imwrite("Result_image_run_" + str(run+1) + ".png", img)
            cv.destroyAllWindows()
            print("\n")
        
        # Plot the camera positions
        plot_camera_positions(tvecs)

        # Store the calibration results
        ret_list.append(ret)
        mtx_list.append(mtx)
        dist_list.append(dist)
        rvecs_list.append(rvecs)
        tvecs_list.append(tvecs)

        print("\n")
        print("Thanks for using the camera calibration tool!")

cv.destroyAllWindows()




