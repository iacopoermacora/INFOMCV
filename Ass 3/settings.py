import cv2 as cv
import json

def read_xml_chekerboard(file_path):
    fs = cv.FileStorage(file_path, cv.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError("Failed to open XML file.")
    
    CheckerBoardWidth = int(fs.getNode("CheckerBoardWidth").real())
    CheckerBoardHeight = int(fs.getNode("CheckerBoardHeight").real())
    CheckerBoardSquareSize = int(fs.getNode("CheckerBoardSquareSize").real())

    fs.release()
    return CheckerBoardSquareSize, CheckerBoardWidth, CheckerBoardHeight

def read_world_dimensions_from_config(filename):
    with open(filename, 'r') as file:
        config = json.load(file)
        world_width = config.get('world_width')
        world_height = config.get('world_height')
        world_depth = config.get('world_depth')
        return world_width, world_height, world_depth

# Parameters Settings
    
WIDTH, HEIGHT, DEPTH = 256, 64, 128 # read_world_dimensions_from_config('config.json') # TODO: try to put manually smaller dimensions (128 o anche meno, 64, 250)

# Set to True to use the webcam for testing, or False to use the test static image (DEFAULT: False)
USE_WEBCAM = False
# Set the error threshold for rejecting low-quality images (DEFAULT: 0.3)
ERROR_THRESHOLD = 0.3
# Define the size of the grid in millimeters
GRID_TILE_SIZE = 30
# Plot the camera positions or not (DEFAULT: False)
PLOT_CAMERA = False
# Show the 3D cube or not
SHOW_CUBE = False
# Show the corners or not
SHOW_CORNERS = False
# Interval of seconds between images (DEFAULT: 10)
INTERVAL = 2
# Number of cameras to calibrate
NUM_CAMERAS = 4
# Time to show the image with auto-detected corners in milliseconds
IMAGE_VIEW_TIME = 100
# Decide whether to create the mesh or not
CREATE_MESH = False
# Read the size of the checkerboard
CHECKERBOARD_SQUARE_SIZE, CHECKERBOARD_WIDTH, CHECKERBOARD_HEIGHT = read_xml_chekerboard("data/checkerboard.xml")
# Block size
BLOCK_SIZE = 1

# Number of frames to analyse per second
NUMBER_OF_FRAMES_TO_ANALYSE = 105 # TODO: Set to 100
# Frame of the video to display in the 3D visualiser
STARTING_FRAME_NUMBER = 0 # NOTE: Starting frame number out of the number of frames to analyse JUST FOR TEST, OTHERWISE SET TO 0
# Maximum number of frames to analyse
MAX_NUMBER_OF_FRAMES = 105 # NOTE: Maximum number of frames to analyse JUST FOR TEST, OTHERWISE SET TO INFINITY
# Index of the frame for offline color model
OFFLINE_IDX = [660, 0, 660, 520] # FIND IDEAL FRAMES
# True if you want to visualize also the real colors of the voxels, False if you want to visualize only the artificially colored clusters
VISUALIZE_REAL_COLORS = False

# Number of clusters
NUMBER_OF_CLUSTERS = 4