import os
import glob
from collections import Counter
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

# HMDB51 DATASET
def create_hmdb51_splits(keep_hmdb51):
    for files in os.listdir('video_data'):
        foldername = files.split('.')[0]
        if foldername in keep_hmdb51:
            if not os.path.exists('video_data/' + foldername): # NOTE: This is to avoid extracting the files again and again.
                # extract only the relevant classes for the assignment.
                os.system("mkdir -p video_data/" + foldername)
                os.system("unrar e video_data/"+ files + " video_data/"+foldername)


    TRAIN_TAG, TEST_TAG = 1, 2
    train_files, test_files = [], []
    train_labels, test_labels = [], []
    split_pattern_name = f"*test_split1.txt"
    split_pattern_path = os.path.join('test_train_splits', split_pattern_name)
    annotation_paths = glob.glob(split_pattern_path)
    for filepath in annotation_paths:
        class_name = '_'.join(filepath.split('/')[-1].split('_')[:-2]) # TODO: Change this line for MAC or Linux
        if class_name not in keep_hmdb51:
            print(f"Skipping {class_name}")
            continue  # skipping the classes that we won't use.
        print(f"Processing {class_name}")
        with open(filepath) as fid:
            lines = fid.readlines()
        for line in lines:
            video_filename, tag_string = line.split()
            tag = int(tag_string)
            if tag == TRAIN_TAG:
                train_files.append(video_filename)
                train_labels.append(class_name)
            elif tag == TEST_TAG:
                test_files.append(video_filename)
                test_labels.append(class_name)

    print(f'Train files ({len(train_files)}):\n\t{train_files}')
    print(f'Train labels ({len(train_labels)}):\n\t{train_labels}\n'\
        f'Train Distribution:{list(Counter(sorted(train_labels)).items())}\n')
    print(f'Test files ({len(test_files)}):\n\t{test_files}')
    print(f'Test labels ({len(test_labels)}):\n\t{test_labels}\n'\
        f'Test Distribution:{list(Counter(sorted(test_labels)).items())}\n')
    action_categories = sorted(list(set(train_labels)))
    print(f'Action categories ({len(action_categories)}):\n{action_categories}')

    return train_files, train_labels, test_files, test_labels

def plot_distribution(train_labels, test_labels, keep_hmdb51):
    # Check if the distribution file already exists
    if os.path.exists("plots/dataset_distributions/hmdb51_distribution.png"):
        print("Distribution file already exists. Skipping the plotting.")
        return
    # Counting items for each class
    train_counter = Counter(train_labels)
    test_counter = Counter(test_labels)

    print("Train Data:")
    for class_name in keep_hmdb51:
        print(f"{class_name}: {train_counter[class_name]}")
    print("\nTest Data:")
    for class_name in keep_hmdb51:
        print(f"{class_name}: {test_counter[class_name]}")

    # Data preparation
    train_data = [train_counter[class_name] for class_name in keep_hmdb51]
    test_data = [test_counter[class_name] for class_name in keep_hmdb51]
    class_names = keep_hmdb51

    # Plotting
    plt.figure(figsize=(12, 6))
    # Sum train_data and test_data to get the total number of videos in each class.
    total_data = [train_data[i] + test_data[i] for i in range(len(train_data))]
    plt.bar(range(len(class_names)), total_data, color='blue', alpha=0.5, label='Train')
    plt.bar(range(len(class_names)), test_data, color='red', alpha=0.5, label='Test')
    plt.xlabel('Action Categories')
    plt.ylabel('Number of Videos')
    plt.title('Distribution of Classes in HMDB51 Dataset')
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/dataset_distributions/hmdb51_distribution.png")

def check_video_length(train_files, train_labels, test_files, test_labels, keep_hmdb51):
    # Check if the video lenght file already exists
    if os.path.exists("plots/dataset_distributions/hmdb51_video_length_distribution.png"):
        print("Video lenght file already exists. Skipping the plotting.")
        return
    train_video_lengths = {class_name: [] for class_name in keep_hmdb51}
    test_video_lengths = {class_name: [] for class_name in keep_hmdb51}

    for video_file, label in zip(train_files, train_labels):
        video_path = os.path.join('video_data', label, video_file)
        cap = cv2.VideoCapture(video_path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        train_video_lengths[label].append(length)

    for video_file, label in zip(test_files, test_labels):
        video_path = os.path.join('video_data', label, video_file)
        cap = cv2.VideoCapture(video_path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        test_video_lengths[label].append(length)

    # Plotting video length distributions
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    for class_name, lengths in train_video_lengths.items():
        plt.hist(lengths, bins=20, alpha=0.5, label=class_name)
    plt.xlabel('Video Length')
    plt.ylabel('Frequency')
    plt.title('Train Video Length Distribution')
    plt.legend()

    plt.subplot(1, 2, 2)
    for class_name, lengths in test_video_lengths.items():
        plt.hist(lengths, bins=20, alpha=0.5, label=class_name)
    plt.xlabel('Video Length')
    plt.ylabel('Frequency')
    plt.title('Test Video Length Distribution')
    plt.legend()

    plt.tight_layout()
    plt.savefig("plots/dataset_distributions/hmdb51_video_length_distribution.png")

def check_frame_size(train_files, train_labels, test_files, test_labels, keep_hmdb51):
    # Check if the frame size file already exists
    if os.path.exists("plots/dataset_distributions/hmdb51_frame_size_distribution.png"):
        print("Frame size file already exists. Skipping the plotting.")
        return
    train_frame_sizes = {class_name: [] for class_name in keep_hmdb51}
    test_frame_sizes = {class_name: [] for class_name in keep_hmdb51}

    for video_file, label in zip(train_files, train_labels):
        video_path = os.path.join('video_data', label, video_file)
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            height, width, _ = frame.shape
            train_frame_sizes[label].append((height, width))

    for video_file, label in zip(test_files, test_labels):
        video_path = os.path.join('video_data', label, video_file)
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            height, width, _ = frame.shape
            test_frame_sizes[label].append((height, width))

    # Plotting frame size distributions
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    for class_name, sizes in train_frame_sizes.items():
        heights, widths = zip(*sizes)
        plt.scatter(widths, heights, alpha=0.5, label=class_name)
    plt.xlabel('Frame Width')
    plt.ylabel('Frame Height')
    plt.title('Train Frame Size Distribution')
    plt.legend()

    plt.subplot(1, 2, 2)
    for class_name, sizes in test_frame_sizes.items():
        heights, widths = zip(*sizes)
        plt.scatter(widths, heights, alpha=0.5, label=class_name)
    plt.xlabel('Frame Width')
    plt.ylabel('Frame Height')
    plt.title('Test Frame Size Distribution')
    plt.legend()

    plt.tight_layout()
    plt.savefig("plots/dataset_distributions/hmdb51_frame_size_distribution.png")

def extract_optical_flow_and_save(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(output_folder, exist_ok=True)

    # Check if it exsists a file in the output folder that starts with the same name as the video file
    if any(file.startswith(os.path.splitext(video_file)[0]) for file in os.listdir(output_folder)):
        print(f"Optical flow images for {video_path} already exist. Skipping the extraction.")
        return
    else:
        print(f"Extracting optical flow images for {video_path}")

    if frame_count < 16:
        print("#" * 200)
        print(f"Video {video_path} has less than 16 frames.")
    step_size = frame_count // 16
    for i, idx in enumerate(range(1, frame_count, step_size)):  # Extract 16 evenly spaced frames
        if i > 15:
            break

        ret, frame1 = cap.read()
        ret, frame2 = cap.read()
        if not ret:
            break

        frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(frame1_gray, frame2_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Calculate magnitude and angle
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Normalize magnitude to range [0, 255] and save as grayscale image
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        # Resize the image to 112x112
        mag_resized = cv2.resize(mag, (112, 112), interpolation=cv2.INTER_AREA)

        cv2.imwrite(os.path.join(output_folder, f"{os.path.splitext(video_file)[0]}_{idx}.png"), mag_resized)

    cap.release()

def extract_frames(video_path, output_folder):
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    
    # Get total number of frames in the video
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Define frame indices to extract
    frame_indices = [0, total_frames // 4, total_frames // 2, (3 * total_frames) // 4, total_frames - 2]
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Extract frames at specified indices
    for idx, frame_idx in enumerate(frame_indices):
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = video_capture.read()
        if success:
            # Save frame to file
            frame_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(output_folder, f"{frame_name}_{idx * 25}.png")
            cv2.imwrite(output_path, frame)
    
    # Release video capture
    video_capture.release()

def process_videos(input_folder, output_root):
    # Iterate through all files and folders in input folder
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.avi'):  # Check if file is a video
                video_path = os.path.join(root, file)
                relative_folder = os.path.relpath(root, input_folder)
                output_folder = os.path.join(output_root, relative_folder)
                extract_frames(video_path, output_folder)

keep_hmdb51 = ["clap", "climb", "drink", "jump", "pour", "ride_bike", "ride_horse", 
            "run", "shoot_bow", "smoke", "throw", "wave"]
train_files, train_labels, test_files, test_labels = create_hmdb51_splits(keep_hmdb51)

# Perform data analysis
plot_distribution(train_labels, test_labels, keep_hmdb51)
check_video_length(train_files, train_labels, test_files, test_labels, keep_hmdb51)
check_frame_size(train_files, train_labels, test_files, test_labels, keep_hmdb51)

# Extract optical flow images
if not os.path.exists("optical_flow_images"):
    # Create optical flow images for training set
    for i, (video_file, video_label) in tqdm(enumerate(zip(train_files, train_labels))):
        video_path = os.path.join("video_data", video_label, video_file)
        output_folder = os.path.join("optical_flow_images", video_label)
        extract_optical_flow_and_save(video_path, output_folder)

    # Create optical flow images for test set
    for i, (video_file, video_label) in tqdm(enumerate(zip(test_files, test_labels))):
        video_path = os.path.join("video_data", video_label, video_file)
        output_folder = os.path.join("optical_flow_images", video_label)
        extract_optical_flow_and_save(video_path, output_folder)

# Extract frames
if not os.path.exists("video_image_dataset"):
    # Process videos
    process_videos("video_data", "video_image_dataset")
