import cv2

def display_specific_frame(video_file, frame_number):
    cap = cv2.VideoCapture(video_file)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Get total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames:", total_frames)

    # Check if the specified frame number is within range
    if frame_number < 0 or frame_number >= total_frames:
        print("Error: Invalid frame number.")
        return

    # Set the frame position to the specified frame number
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    ret, frame = cap.read()

    # Check if frame is read successfully
    if not ret:
        print("Error: Could not read frame.")
        return

    # Display the frame
    cv2.imshow("Frame", frame)
    cv2.waitKey(0)

    # Release the video capture object and close the window
    cap.release()
    cv2.destroyAllWindows()

# Example usage
video_file = "data/cam4/video.avi"
frame_number = 520 # Specify the frame number you want to display
display_specific_frame(video_file, frame_number)
