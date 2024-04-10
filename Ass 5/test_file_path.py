import os
import re

flow_folder_path = 'video_OF_dataset/clap'
file_name = '_Boom_Snap_Clap__challenge_clap_u_nm_np1_fr_med_0.avi'
print(os.path.splitext(file_name)[0])
image_files = [file for file in os.listdir(flow_folder_path) if file.startswith(os.path.splitext(file_name)[0])]
def sort_by_number(filename):
    return [int(x) if x.isdigit() else x for x in re.findall(r'\d+|\D+', filename)]
# Sort the list using the custom sorting function
image_files = sorted(image_files, key=sort_by_number)
print(image_files)