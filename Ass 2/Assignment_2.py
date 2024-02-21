import camera_calibration.py as cc
import settings.py as settings

# Call the function to get the camera intrinsics and extrinsics for each camera
for camera_number in range(1, num_cameras+1):
    cc.get_camera_intrinsics_and_extrinsics(camera_number)