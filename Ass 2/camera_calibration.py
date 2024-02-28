import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
import os
import settings as settings

def find_corners(fname, img, gray, objp, objpoints, imgpoints, criteria):
    global height
    global width
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (settings.CHECKERBOARD_HEIGHT, settings.CHECKERBOARD_WIDTH), None, cv.CALIB_CB_FAST_CHECK)
    # If found corners, add object points, image points (after refining them) otherwise manual select them
    if ret == True:
        print("\tAuto-detect corners: ", fname)
        subpix = True
    else:
        if fname.split('/')[-1] == "Test_image.jpg":
            print("\tUnfortunately, the corners were not found automatically. Please select them manually.")
            print("\tManually select corners: ", fname)
            corners = manual_corners_selection(gray, img)
            subpix = False
    
    if (fname.split('/')[-1] == "Test_image.jpg") or (ret == True): # TODO: Remove this condition, it is only to speed up testing
        corners = add_corners_show_image(gray, img, corners, objp, objpoints, imgpoints, criteria, subpix)

    return corners

def click_event(event, x, y, flags, params):
    '''
    click_event: Function to handle mouse clicks on the image

    :param event: The event type
    :param x: The x-coordinate of the mouse click
    :param y: The y-coordinate of the mouse click
    :param flags: The flags (not used in this assignment)
    :param params: The parameters (img, manual_coordinates)
    '''
    img, manual_coordinates, click = params

    # Checking for left mouse clicks 
    if event == cv.EVENT_LBUTTONDOWN: 
        if click[0] < 4:  # Ensure only 4 points are selected
            manual_coordinates[click] = (x, y)
            click[0] += 1
            print("click", click[0])
            # Draw a small circle at the clicked point
            cv.circle(img, (x, y), 3, (0, 0, 255), -1)
            cv.namedWindow('img', cv.WINDOW_NORMAL)
            cv.imshow('img', img)
            print("\tCorner found at: ", x, y)
        if click[0] == 4:
            # font 
            font = cv.FONT_HERSHEY_SIMPLEX
            # org 
            org = (10, 50) 
            # fontScale 
            fontScale = 0.75
            # Blue color in BGR
            color = (255, 0, 0) 
            # Line thickness of 2 px
            thickness = 3
            cv.putText(img, "Press any key to find all chessboard corners", org, font, fontScale, color, thickness, cv.LINE_AA)
            cv.namedWindow('img', cv.WINDOW_NORMAL)
            cv.imshow('img', img)

def manual_corners_selection(gray, img):
    '''
    manual_corners_selection: Function to manually select the corners of the chessboard

    :param gray: The grayscale image of the chessboard
    :param img: The original image of the chessboard

    :return: None (but save the four corners of the image in the global variable corners)
    '''

    print("\tSelect 4 corners in the image in the order: top-left, top-right, bottom-right, bottom-left.")
    print("\tThe corners should be selected in a clockwise order and the first selected side should be long ", settings.CHECKERBOARD_WIDTH-1, " squares.")
    input("\tPress Enter to continue...")
    print("\tClick on the image to select the corners...")

    manual_coordinates = np.zeros((4, 2), dtype=np.float32)
    click = [0]

    # Display the image and wait for the user to select the corners
    cv.namedWindow('img', cv.WINDOW_NORMAL)
    cv.imshow('img', img)
    params = [img, manual_coordinates, click]
    cv.setMouseCallback('img', click_event, params) # Limit to 4 clicks
    cv.waitKey(0) # Press any key to continue after 4 clicks
    cv.destroyAllWindows()
    cv.waitKey(1)

    # Perspective transformation
    target_height, target_width = img.shape[:2]  # You can set a specific size
    new_corners = np.float32([[0, 0], [target_width, 0], [target_width, target_height], [0, target_height]])
    # Calculate perspective transform matrix
    matrix = cv.getPerspectiveTransform(manual_coordinates, new_corners)
    # Warp perspective points
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
    
    # Define the grid coordinates in the warped image
    grid_points = np.zeros(((settings.CHECKERBOARD_HEIGHT) * (settings.CHECKERBOARD_WIDTH), 2), dtype=np.float32)

    # Generate grid coordinates
    index = 0
    for j in range(settings.CHECKERBOARD_WIDTH):
        for i in range(settings.CHECKERBOARD_HEIGHT-1, -1, -1):
            x = j * width_warp/(settings.CHECKERBOARD_WIDTH-1)
            y = i * height_warp/(settings.CHECKERBOARD_HEIGHT-1)
            grid_points[index] = (x, y)
            index += 1

    # Calculate the inverse of the matrix
    matrix_inv = cv.invert(matrix)[1]
    # Calculate the corners in the original image
    corners = cv.perspectiveTransform(grid_points.reshape(-1, 1, 2), matrix_inv)

    return corners

def add_corners_show_image(gray, img, corners, objp, objpoints, imgpoints, criteria, subpix=True):
    '''
    add_corners_show_image: Function to add all the (inner) corners to the image and display it

    :param gray: The grayscale image of the chessboard
    :param img: The original image of the chessboard

    :return: None (but display the image with the corners)
    '''

    # Save the object points
    objpoints.append(objp)
    if(subpix):
        # Improve the corner positions
        corners = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        # Save the image points
        imgpoints.append(corners)
    else:
        imgpoints.append(corners)

    # Draw and display the corners
    cv.namedWindow('img', cv.WINDOW_NORMAL)
    if subpix:
        cv.drawChessboardCorners(img, (settings.CHECKERBOARD_HEIGHT, settings.CHECKERBOARD_WIDTH), corners, True)
    cv.imshow('img', img)
    cv.waitKey(settings.IMAGE_VIEW_TIME)
    cv.destroyAllWindows()
    cv.waitKey(1)

    return corners

def process_frame(fname, img, objp, objpoints, imgpoints, criteria, mtx, dist, axis, cube_points):
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

    corners_test = find_corners(fname, img, gray, objp, objpoints, imgpoints, criteria)

    # Find the rotation and translation vectors.
    ret, rvecs, tvecs = cv.solvePnP(objp, corners_test, mtx, dist)
    imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
    imgpts_cube, jac = cv.projectPoints(cube_points, rvecs, tvecs, mtx, dist)
    img = draw(img, corners_test, imgpts)
    if settings.SHOW_CUBE:
        img = draw_cube(img, imgpts_cube, color=(255, 255, 0), thickness=1)
    return img, rvecs, tvecs

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

def reject_low_quality_images(ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints, image_size):
    '''
    reject_low_quality_images: Function to reject low-quality images and recalibrate

    :param ret: The return value from the calibration
    :param mtx: The camera matrix
    :param dist: The distortion coefficients
    :param rvecs: The rotation vectors
    :param tvecs: The translation vectors

    :return: The return value from the calibration, the camera matrix, the distortion coefficients, the rotation vectors, and the translation vectors
    '''

    print("Rejecting low-quality images...")
    mean_error = validate_calibration(objpoints, imgpoints, rvecs, tvecs, mtx, dist)
    print(f"\tMean reprojection error: {mean_error}")

    # Iterate to reject low-quality images
    while mean_error > settings.ERROR_THRESHOLD and len(objpoints) > 0:
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
        if max_error_value > settings.ERROR_THRESHOLD:
            print(f"\tRejecting image with error {max_error_value}")
            objpoints.pop(max_error_idx)
            imgpoints.pop(max_error_idx)
            # Re-calibrate without the worst image
            ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image_size, None, None)

            # Recalculate mean error
            mean_error = validate_calibration(objpoints, imgpoints, rvecs, tvecs, mtx, dist)
            print(f"\tMean validation error after rejection: {mean_error}")
        else:
            print(f"\tNo more images to reject with error above {settings.ERROR_THRESHOLD}")
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
    N = settings.CHECKERBOARD_SQUARE_SIZE * (settings.CHECKERBOARD_WIDTH-1)
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

    img = cv.line(img, corner, tupleOfInts(imgpts[0].ravel()), (255,0,0), 1)
    img = cv.line(img, corner, tupleOfInts(imgpts[1].ravel()), (0,255,0), 1)
    img = cv.line(img, corner, tupleOfInts(imgpts[2].ravel()), (0,0,255), 1)
    return img

def draw_cube(img, imgpts_cube, color=(0, 255, 0), thickness=1):
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

def get_images_from_video(camera_number, video_path, test_image=False):
    # Open the video
    cap = cv.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video file")

    # Get the frame rate of the video
    fps = cap.get(cv.CAP_PROP_FPS)

    # Calculate the interval between each frame to capture (every 2 seconds)
    interval = settings.INTERVAL * fps

    count = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # If frame_count modulo interval is 0, save the frame
            if frame_count % interval == 0:
                if test_image == True:
                    frame_name = f'data/cam{camera_number}/Test_image.jpg'
                else:
                    frame_name = f'data/cam{camera_number}/frames/frame_{count}.jpg'
                cv.imwrite(frame_name, frame)
                count += 1
            frame_count += 1
            if test_image == True:
                break
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv.destroyAllWindows()

def print_explicit_intrinsics(mtx, test_image):

    print("Intrinsic camera matrix: \n\t", mtx)

    # Assume mtx, rvecs, tvecs are obtained from cv.calibrateCamera()
    focal_length_x = mtx[0, 0]
    focal_length_y = mtx[1, 1]
    principal_point_x = mtx[0, 2]
    principal_point_y = mtx[1, 2]
    aspect_ratio = focal_length_x / focal_length_y
    # Define the image properties (based on the test image as all the images come from the same camera)
    info_image = cv.imread(test_image)
    # Image resolution (width, height)
    height_image, width_image, _ = info_image.shape
    image_resolution = (width_image, height_image)

    print(f"\tFocal length in x (f_x): {focal_length_x}")
    print(f"\tFocal length in y (f_y): {focal_length_y}")
    print(f"\tPrincipal point x (c_x): {principal_point_x}")
    print(f"\tPrincipal point y (c_y): {principal_point_y}")
    print(f"\tAspect ratio: {aspect_ratio}")
    print(f"\tResolution of the images: {image_resolution}")
    print("\n")
    
def write_camera_parameters(camera_number, camera_matrix, dist_coeffs, rvecs, tvecs):

    directory = f'data/cam{camera_number}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Define the file path for the config file
    file_name = f"{directory}/config.xml"
    
    # Open a FileStorage object to write data to the XML file
    fs = cv.FileStorage(file_name, cv.FILE_STORAGE_WRITE)

    fs.write("camera_matrix", camera_matrix)
    fs.write("distortion_coefficients", dist_coeffs)
    fs.write("rotation_vectors", np.asarray(rvecs, dtype=np.float32))
    fs.write("translation_vectors", np.asarray(tvecs, dtype=np.float32))

    fs.release()

def get_camera_intrinsics_and_extrinsics(camera_number):
    
    if os.path.exists(f'data/cam{camera_number}/config.xml'):
        return
    
    # If the test image is not found, get it from the video
    if not os.path.exists(f'data/cam{camera_number}/Test_image.jpg'):
        get_images_from_video(camera_number, f'data/cam{camera_number}/checkerboard.avi', test_image=True)
    # Define the test image to use for the static image test
    test_image = f'data/cam{camera_number}/Test_image.jpg'

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points
    objp = np.zeros(((settings.CHECKERBOARD_WIDTH)*(settings.CHECKERBOARD_HEIGHT), 3), np.float32)
    objp[:,:2] = np.mgrid[0:(settings.CHECKERBOARD_HEIGHT),0:(settings.CHECKERBOARD_WIDTH)].T.reshape(-1,2) * settings.CHECKERBOARD_SQUARE_SIZE # Adjusted to the grid real dimensions
    # Variable to store the image size
    image_size = None

    if not os.path.exists(f'data/cam{camera_number}/frames'):
        os.makedirs(f'data/cam{camera_number}/frames')
        
    images = glob.glob(f'data/cam{camera_number}/frames/*.jpg')
    # If the images are not found, get them from the video
    if not(images):
        print("Images not found. Getting images from video...")
        get_images_from_video(camera_number, f'data/cam{camera_number}/intrinsics.avi')
        images = glob.glob(f'data/cam{camera_number}/frames/*.jpg')

    print("\n\n")
    print("-" * 50) # Print a separator

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # Reset the counter for the auto-found corners
    counter_auto_found = 0

    for fname in images:
        print("Processing image: ", fname)
        # Read the image and convert to grayscale
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        find_corners(fname, img, gray, objp, objpoints, imgpoints, criteria)

        if image_size is None:
            image_size = gray.shape[::-1]

        print("\n")
        
    if image_size is not None:
        
        # Calibrate the camera
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image_size, None, None)

        # Reject low-quality images and recalibrate
        ret, mtx, dist, rvecs, tvecs = reject_low_quality_images(ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints, image_size)

        print_explicit_intrinsics(mtx, test_image)

        # Define the object points for the test image
        objp_test = np.zeros(((settings.CHECKERBOARD_WIDTH)*(settings.CHECKERBOARD_HEIGHT),3), np.float32)
        objp_test[:,:2] = np.mgrid[0:(settings.CHECKERBOARD_HEIGHT),0:(settings.CHECKERBOARD_WIDTH)].T.reshape(-1,2) * settings.CHECKERBOARD_SQUARE_SIZE
        # Define the axis
        axis = np.float32([[4,0,0], [0,4,0], [0,0,-4]]).reshape(-1,3) * settings.CHECKERBOARD_SQUARE_SIZE
        # Define the cube points
        cube_size = 2
        cube_points = np.float32([
        [0, 0, 0], [0, cube_size, 0], [cube_size, cube_size, 0], [cube_size, 0, 0],
        [0, 0, -cube_size], [0, cube_size, -cube_size], [cube_size, cube_size, -cube_size], [cube_size, 0, -cube_size]
        ]) * settings.CHECKERBOARD_SQUARE_SIZE

        # Test the calibration using the webcam or the static test image
        if settings.USE_WEBCAM:
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
                frame, rvecs_test, tvecs_test = process_frame("Webcam frame", frame, objp_test, objpoints, imgpoints, criteria, mtx, dist, axis, cube_points)
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
            img, rvecs_test, tvecs_test = process_frame(test_image, img, objp_test, objpoints, imgpoints, criteria, mtx, dist, axis, cube_points)
            cv.namedWindow('img', cv.WINDOW_NORMAL)
            cv.imshow('img', img)
            # Save the image if 's' is pressed
            k = cv.waitKey(0) & 0xFF
            if k == ord('s'):
                cv.imwrite("Result_image.png", img)
            cv.destroyAllWindows()
            print("\n")
        
        cv.waitKey(1)
        
        write_camera_parameters(camera_number, mtx, dist, rvecs_test, tvecs_test)
        
        if settings.PLOT_CAMERA:
            # Plot the camera positions
            plot_camera_positions(tvecs)

        print("\n")
    cv.destroyAllWindows()
    cv.waitKey(1)
