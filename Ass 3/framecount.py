import cv2

def count_frames(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if the video file is opened successfully
    if not video.isOpened():
        print("Error: Couldn't open the video file.")
        return -1

    # Initialize frame count
    frame_count = 0

    # Loop through the video frames
    while True:
        # Read the next frame
        ret, frame = video.read()

        # Check if the frame is read correctly
        if not ret:
            break

        # Increment frame count
        frame_count += 1

    # Release the video object
    video.release()

    return frame_count

# Path to your video file
video_path = 'data/cam1/video.avi'

# Count frames
total_frames = count_frames(video_path)
print("Total frames in the video:", total_frames)
