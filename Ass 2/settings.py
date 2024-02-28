import cv2 as cv

def read_xml_chekerboard(file_path):
    fs = cv.FileStorage(file_path, cv.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError("Failed to open XML file.")
    
    CheckerBoardWidth = int(fs.getNode("CheckerBoardWidth").real())
    CheckerBoardHeight = int(fs.getNode("CheckerBoardHeight").real())
    CheckerBoardSquareSize = int(fs.getNode("CheckerBoardSquareSize").real())

    fs.release()
    return CheckerBoardSquareSize, CheckerBoardWidth, CheckerBoardHeight

# Parameters Settings

# Set to True to use the webcam for testing, or False to use the test static image (DEFAULT: False)
USE_WEBCAM = False
# Set the error threshold for rejecting low-quality images (DEFAULT: 0.3)
ERROR_THRESHOLD = 0.5 
# Define the width and height of the internal chessboard (in squares)
# width = 5 # TODO: In checkerboard.xml width and height are switched ???
# height = 7
# Define the size of the squares of the chessboard in millimeters TODO: I think a problem could be here!
# square_size = 115
# Define the size of the grid in millimeters
GRID_TILE_SIZE = 15
# Plot the camera positions or not (DEFAULT: False)
PLOT_CAMERA = False
# Interval of seconds between images (DEFAULT: 10)
INTERVAL = 10
# Number of cameras to calibrate
NUM_CAMERAS = 4
# Time to show the image with auto-detected corners in milliseconds
IMAGE_VIEW_TIME = 100
# Frame of the video to display in the 3D visualiser
FRAME_NUMBER = 0

CHECKERBOARD_SQUARE_SIZE, CHECKERBOARD_WIDTH, CHECKERBOARD_HEIGHT = read_xml_chekerboard("data/checkerboard.xml")