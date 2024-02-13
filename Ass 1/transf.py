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
            if click == 4:  # After 4 points, close the window
                cv.destroyAllWindows()

images = glob.glob('*.jpg')
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

    # Display the result
    cv.namedWindow('Warped Image', cv.WINDOW_NORMAL)
    cv.imshow('Warped Image', warped_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Reset click count and manual_coordinates for the next image
    click = 0
    manual_coordinates = np.zeros((4, 2), dtype=np.float32)
