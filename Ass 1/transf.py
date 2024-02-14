import numpy as np
import cv2 as cv
import glob

manual_coordinates = np.zeros((4, 2), dtype=np.float32)
click = 0

rows = 8  # Number of rows in the grid
cols = 5  # Number of columns in the grid

def click_event(event, x, y, flags, params): 
    global click
    global manual_coordinates

    # Checking for left mouse clicks 
    if event == cv.EVENT_LBUTTONDOWN: 
        if click < 4:  # Ensure only 4 points are selected
            manual_coordinates[click] = (x, y)
            click += 1
            print("Corner found at: ", x, y)

images = glob.glob('*.JPG')
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    h, w = img.shape[:2]

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

    # Perspective transformation
    target_width, target_height = w, h  # You can set a specific size
    new_corners = np.float32([[0, 0], [target_width, 0], [target_width, target_height], [0, target_height]])
    # Calculate perspective transform matrix
    matrix = cv.getPerspectiveTransform(manual_coordinates, new_corners)
    # Apply perspective transform
    warped_image = cv.warpPerspective(img, matrix, (target_width, target_height))

    warped_points = cv.perspectiveTransform(manual_coordinates.reshape(-1, 1, 2), matrix)

    print(warped_points)

    # Initialize variables to store the width and height of the warped image
    width = 0
    height = 0

    # Iterate through warped_points to find width and height
    for point in warped_points:
        x, y = point[0]
        if y > height:
            height = y
        if x > width:
            width = x
        cv.circle(warped_image, (int(x), int(y)), 30, (0, 255, 0), 20) # Draw a circle around the points

    print("Width: ", width)
    print("Height: ", height)
    
    # Define the grid coordinates
    grid_points = np.zeros(((rows+1) * (cols+1), 2), dtype=np.float32)

    # Generate grid coordinates
    index = 0
    for i in range(rows+1):
        for j in range(cols+1):
            x = j * width/cols
            y = i * height/rows
            grid_points[index] = (x, y)  # Adjust 100 according to your grid spacing
            index += 1
            cv.circle(warped_image, (int(x), int(y)), 10, (0, 255, 0), 5) # Draw a circle around the points

    # Display the result
    cv.namedWindow('Warped Image', cv.WINDOW_NORMAL)
    cv.imshow('Warped Image', warped_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    Minv = cv.invert(matrix)[1]
    original_points = cv.perspectiveTransform(grid_points.reshape(-1, 1, 2), Minv)

    for point in original_points:
        x, y = point[0]
        cv.circle(img, (int(x), int(y)), 30, (0, 255, 0), 20) # Draw a circle around the points
    
    cv.namedWindow('img', cv.WINDOW_NORMAL)
    cv.imshow('img', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Reset click count and manual_coordinates for the next image
    click = 0
    manual_coordinates = np.zeros((4, 2), dtype=np.float32)
