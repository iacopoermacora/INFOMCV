from collections import Counter
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import random
from PIL import Image

def create_stanford40_splits():
    # STANFORD 40 DATASET
    keep_stanford40 = ["applauding", "climbing", "drinking", "jumping", "pouring_liquid", "riding_a_bike", "riding_a_horse", 
            "running", "shooting_an_arrow", "smoking", "throwing_frisby", "waving_hands"]
    with open('Stanford40/ImageSplits/train.txt', 'r') as f:
        # We won't use these splits but split them ourselves
        train_files = [file_name for file_name in list(map(str.strip, f.readlines())) if '_'.join(file_name.split('_')[:-1]) in keep_stanford40]
        train_labels = ['_'.join(name.split('_')[:-1]) for name in train_files]

    with open('Stanford40/ImageSplits/test.txt', 'r') as f:
        # We won't use these splits but split them ourselves
        test_files = [file_name for file_name in list(map(str.strip, f.readlines())) if '_'.join(file_name.split('_')[:-1]) in keep_stanford40]
        test_labels = ['_'.join(name.split('_')[:-1]) for name in test_files]

    # Combine the splits and split for keeping more images in the training set than the test set.
    all_files = train_files + test_files
    all_labels = train_labels + test_labels
    train_files, test_files = train_test_split(all_files, test_size=0.1, random_state=0, stratify=all_labels)
    train_labels = ['_'.join(name.split('_')[:-1]) for name in train_files]
    test_labels = ['_'.join(name.split('_')[:-1]) for name in test_files]
    print(f'Train files ({len(train_files)}):\n\t{train_files}')
    print(f'Train labels ({len(train_labels)}):\n\t{train_labels}\n'\
        f'Train Distribution:{list(Counter(sorted(train_labels)).items())}\n')
    print(f'Test files ({len(test_files)}):\n\t{test_files}')
    print(f'Test labels ({len(test_labels)}):\n\t{test_labels}\n'\
        f'Test Distribution:{list(Counter(sorted(test_labels)).items())}\n')
    action_categories = sorted(list(set(train_labels)))
    print(f'Action categories ({len(action_categories)}):\n{action_categories}')

    return train_files, train_labels, test_files, test_labels

def check_distribution(train_files, train_labels, test_files, test_labels):
    # Check if the lenght of the train and test files are the same as the labels
    assert len(train_files) == len(train_labels), "Train files and labels are not the same length"
    assert len(test_files) == len(test_labels), "Test files and labels are not the same length"
    # Count items for each of the 12 classes
    train_distribution = dict(Counter(sorted(train_labels)))
    test_distribution = dict(Counter(sorted(test_labels)))
    print("\n\n")
    print("Train Distribution:")
    for label, count in train_distribution.items():
        print(f"{label}: {count}")
    print("\nTest Distribution:")
    for label, count in test_distribution.items():
        print(f"{label}: {count}")
    return train_distribution, test_distribution

def plot_distribution(train_distribution, test_distribution, name):
    if os.path.exists(f"plots/dataset_distributions/{name}.png"):
        print(f"Dataset distribution plot {name} already exists. Skipping...")
        return
    
    if not os.path.exists("plots/dataset_distributions"):
        os.makedirs("plots/dataset_distributions")

    # Convert distributions to lists for plotting
    train_labels_list, train_counts = zip(*train_distribution.items())
    test_labels_list, test_counts = zip(*test_distribution.items())

    # Plotting
    plt.figure(figsize=(12, 6))
    # Sum train_data and test_data to get the total number of images in each class.
    total_data = [train_counts[i] + test_counts[i] for i in range(len(train_counts))]
    total_count = sum(total_data)

    # Plotting bars
    plt.bar(test_labels_list, total_data, color='red', alpha=0.7, label='Test')
    plt.bar(train_labels_list, train_counts, color='blue', alpha=0.7, label='Train')

    # Adding percentages
    for i, count in enumerate(total_data):
        percentage = (count / total_count) * 100
        plt.text(i, count + 5, f'{percentage:.2f}%', ha='center')

    # Plot labels and title
    plt.xlabel('Action Classes')
    plt.ylabel('Count')
    plt.title('Distribution of Action Classes in Train and Test Sets')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/dataset_distributions/{name}.png")

def show_image(image_no, train_files, train_labels):
    # Change image_no to a number between [0, 1200] and you can see a different training image
    img = cv2.imread(f'Stanford40/JPEGImages/{train_files[image_no]}')
    print(f'An image with the label - {train_labels[image_no]}')
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to check image sizes
def check_image_sizes(file_paths):
    sizes = set()
    for file_path in file_paths:
        img = cv2.imread(file_path)
        sizes.add(img.shape[:2])  # Considering only height and width
    return sizes

# Function to check image sizes and find the smallest image
def find_smallest_image(file_paths):
    smallest_height = float('inf')
    smallest_width = float('inf')
    smallest_image_path = None
    for file_path in file_paths:
        img = cv2.imread(file_path)
        if img is not None:
            height, width, _ = img.shape
            if height < smallest_height:
                smallest_height = height
                smallest_image_path = None
            if width < smallest_width:
                smallest_width = width
                smallest_image_path = None
            if height == smallest_height and width == smallest_width:
                smallest_image_path = file_path
    return smallest_image_path, smallest_height, smallest_width

def get_image_dimensions(file_paths):
        dimensions = []
        for file_path in file_paths:
            img = cv2.imread(file_path)
            if img is not None:
                height, width, _ = img.shape
                dimensions.append((height, width))
        return dimensions

def plot_image_dimensions_distribution(train_file_paths, test_file_paths):
    if os.path.exists("plots/dataset_distributions/stanford40_frame_size_distribution.png"):
        print("Image size distribution plot already exists. Skipping...")
        return
    
    if not os.path.exists("plots/dataset_distributions"):
        os.makedirs("plots/dataset_distributions")

    # Directory containing images
    all_file_paths = train_file_paths + test_file_paths

    # Get image dimensions
    all_dimensions = get_image_dimensions(all_file_paths)

    # Count occurrences of each dimension
    dimension_counts = {}
    for dimension in all_dimensions:
        dimension_counts[dimension] = dimension_counts.get(dimension, 0) + 1

    # Prepare data for 3D plot
    heights, widths = zip(*dimension_counts.keys())
    occurrences = list(dimension_counts.values())

    # Plotting
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create bar plot
    ax.bar3d(heights, widths, np.zeros_like(occurrences), 1, 1, occurrences, color='skyblue')

    # Set labels and title
    ax.set_xlabel('Height')
    ax.set_ylabel('Width')
    ax.set_zlabel('Frequency')
    ax.set_title('Distribution of Image Dimensions')

    plt.savefig("plots/dataset_distributions/stanford40_frame_size_distribution.png") 

# Function to resize images and save them to a new directory
def resize_and_save_images(input_folder, output_folder, file_list, target_size=(224, 224)):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        print("Output folder already exists. Skipping resizing.")
        return
    
    # Loop through each file in the file list
    for filename in file_list:
        # Read the image
        img = cv2.imread(os.path.join(input_folder, filename))
        
        # Resize the image
        resized_img = cv2.resize(img, target_size)
        
        # Save the resized image to the output folder
        cv2.imwrite(os.path.join(output_folder, filename), resized_img)

def augment_image_randomly(image_path):
    """Apply a randomly selected augmentation (crop, rotate, or color change) with higher dynamic parameters to an image."""
    image = Image.open(image_path)
    image_np = np.array(image)

    # Define augmentations with higher degrees
    augmentations = [
        iaa.Crop(percent=(0, 0.3)),  # Larger random crops
        iaa.Affine(rotate=iap.Uniform(-360, 360)),  # Higher range for rotation, between -90 and 90 degrees
        iaa.AddToHueAndSaturation(iap.Uniform(-200, 200)),  # Greater adjustments to hue and saturation
        iaa.Multiply((0.5, 1.5)),  # More significant changes in brightness
    ]

    # Randomly select one of the augmentations
    augmentation = random.choice(augmentations)
    image_aug = augmentation.augment_image(image_np)
    
    return Image.fromarray(image_aug)


def balance_dataset(train_files, train_labels, dataset_path):
    # Count occurrences of each class
    label_counts = Counter(train_labels)
    max_count = max(label_counts.values())

    # Check if the dataset has been already augmented
    if os.path.exists('augmented_files.txt') and os.path.exists('augmented_labels.txt'):
        print("Dataset already augmented. Skipping...")
        with open('augmented_files.txt', 'r') as f:
            train_files_augmented = f.read().splitlines()
        with open('augmented_labels.txt', 'r') as f:
            train_labels_augmented = f.read().splitlines()
        return train_files_augmented, train_labels_augmented

    # Initialize lists for augmented data
    train_files_augmented = []
    train_labels_augmented = []

    # To keep track of which images have been augmented
    already_augmented = set()

    for label, count in label_counts.items():
        if count < max_count:
            augmentations_needed = max_count - count
            files_for_label = [f for f, l in zip(train_files, train_labels) if l == label]

            for file_path in files_for_label:
                if augmentations_needed == 0:
                    break

                # Check if this file has already been augmented
                if file_path in already_augmented:
                    continue  # Skip to the next file

                # Perform augmentation
                augmented_image = augment_image_randomly(os.path.join(dataset_path, file_path))

                # Construct a new filename for the augmented image
                base, extension = os.path.splitext(file_path)
                aug_file_name = f"{base}_augmented{extension}"
                aug_full_path = os.path.join(dataset_path, aug_file_name)

                # Save the augmented image
                augmented_image.save(aug_full_path)

                # Update the augmented lists
                train_files_augmented.append(aug_file_name)  # Use full path if needed
                train_labels_augmented.append(label)

                # Mark this image as augmented
                already_augmented.add(file_path)

                augmentations_needed -= 1


    #save in two text files the augmented files and labels
    with open('augmented_files.txt', 'w') as f:
        for item in train_files_augmented:
            f.write("%s\n" % item)
    with open('augmented_labels.txt', 'w') as f:
        for item in train_labels_augmented:
            f.write("%s\n" % item)
                      
    return train_files_augmented, train_labels_augmented


train_files, train_labels, test_files, test_labels = create_stanford40_splits()

# Perform data and distribution analysis

train_distribution, test_distribution = check_distribution(train_files, train_labels, test_files, test_labels)
plot_distribution(train_distribution, test_distribution, "standfor40_class_distribution")
'''show_image(234, train_files, train_labels)'''

# Directory containing images
image_dir = 'Stanford40/JPEGImages/'
train_file_paths = [os.path.join(image_dir, file_name) for file_name in train_files]
test_file_paths = [os.path.join(image_dir, file_name) for file_name in test_files]

# Image size analysis
train_image_sizes = check_image_sizes(train_file_paths)
test_image_sizes = check_image_sizes(test_file_paths)

print("Train Image Sizes:", train_image_sizes)
print("Test Image Sizes:", test_image_sizes)

all_file_paths = train_file_paths + test_file_paths

# Find smallest image in the dataset
smallest_image_path, smallest_height, smallest_width = find_smallest_image(all_file_paths)

print("Smallest Image Path:", smallest_image_path)
print("Smallest Image Size:", smallest_height, smallest_width)

plot_image_dimensions_distribution(train_file_paths, test_file_paths)

# Paths to the input and output folders
input_folder = "Stanford40/JPEGImages"
output_folder_train = "photo_dataset/train"
output_folder_test = "photo_dataset/test"

# Resize and save train images
resize_and_save_images(input_folder, output_folder_train, train_files)

# Resize and save test images
resize_and_save_images(input_folder, output_folder_test, test_files)

# Balance the dataset TODO: is there a check that do not augment if it was already augmented?
train_files_augmented, train_labels_augmented = balance_dataset(train_files, train_labels, "photo_dataset/train")

# Check the distribution of the augmented dataset
# TODO: concatenate the augmented files with the original files
print("Distribution after augmentation: ")
train_distribution_augmented, _ = check_distribution(train_files_augmented + train_files, train_labels_augmented + train_labels, test_files, test_labels)
#print the size of the augmented dataset
print(f"Size of the augmented dataset: {len(train_files_augmented)}")
plot_distribution(train_distribution_augmented, test_distribution, "standfor40_augmented_class_distribution")


