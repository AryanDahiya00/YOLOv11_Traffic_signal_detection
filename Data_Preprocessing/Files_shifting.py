import os
import shutil

def move_image_files(source_folder, destination_folder):
    # List of common image file extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    
    # Check if the destination folder exists, create it if it doesn't
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # Walk through all subdirectories in the source folder
    for dirpath, _, filenames in os.walk(source_folder):
        for filename in filenames:
            # Check if the file is an image file
            if filename.lower().endswith(image_extensions):
                # Construct full file path
                file_path = os.path.join(dirpath, filename)
                # Move the file to the destination folder
                shutil.move(file_path, destination_folder)
                print(f'Moved: {file_path} to {destination_folder}')

# Example usage
source_folder = 'images'  # Replace with your source folder path
destination_folder = 'Images_sorted'  # Replace with your destination folder path

move_image_files(source_folder, destination_folder)
