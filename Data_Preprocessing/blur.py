import cv2
import os
import glob
import random
import shutil

def blur_images_and_save(input_folder, output_folder, blur_percentage):
    """
    Blurs 50% of images in the input folder and saves them along with their 
    corresponding annotation files to the output folder.

    Args:
        input_folder (str): Path to the folder containing original images and annotation files.
        output_folder (str): Path to the folder where blurred images and annotations will be saved.
        blur_percentage (int): The percentage of the image dimensions to calculate blur kernel size.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image paths in the input folder
    image_paths = glob.glob(os.path.join(input_folder, "*.jpg"))
    
    # Randomly select 50% of the images
    selected_images = random.sample(image_paths, int(len(image_paths) * 0.7))

    
    for img_path in selected_images:
        # Load the image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Skipping invalid image: {img_path}")
            continue
        
        # Get image dimensions
        h, w = image.shape[:2]
        
        # Compute kernel size for blur (must be odd)
        kernel_size = max(1, int(min(h, w) * blur_percentage / 100))
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        
        # Apply Gaussian blur
        blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        # Save blurred image to output folder
        image_name = os.path.basename(img_path)
        blurred_image_path = os.path.join(output_folder, image_name)
        cv2.imwrite(blurred_image_path, blurred_image)
        
        # Copy corresponding annotation file
        txt_path = img_path.replace(".jpg", ".txt")
        if os.path.exists(txt_path):
            shutil.copy(txt_path, os.path.join(output_folder, os.path.basename(txt_path)))

# Paths for train and test folders
train_folder = "train"  # Replace with your train folder path
test_folder = "test"    # Replace with your test folder path

# Create main output folder and subfolders
main_output_folder = "new"
blurred_train_folder = os.path.join(main_output_folder, "blurred_train")
blurred_test_folder = os.path.join(main_output_folder, "blurred_test")

# Apply blur and save for both train and test datasets
blur_percentage = 3
blur_images_and_save(train_folder, blurred_train_folder, blur_percentage)
blur_images_and_save(test_folder, blurred_test_folder, blur_percentage)

print(f"Blurring and saving completed. Check the folder: {main_output_folder}")