import cv2
import matplotlib.pyplot as plt

# Function to resize and visualize images
def resize_and_visualize(image_path, target_size):
    # Read the original image
    original_image = cv2.imread(image_path)
    
    # Resize the image
    resized_image = cv2.resize(original_image, target_size)
    
    # Visualize original and resized images
    plt.figure(figsize=(10, 5))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    # Resized image
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    plt.title('Resized Image')
    plt.axis('off')
    
    plt.show()

# Example usage
image_path = 'Stanford40/JPEGImages/applauding_005.jpg'  # Specify the path to your image
target_size = (112, 112)  # Specify the target size for resizing

resize_and_visualize(image_path, target_size)
