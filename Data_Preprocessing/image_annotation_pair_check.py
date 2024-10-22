import os

def match_and_clean_pairs(image_dir, xml_dir):
    # Get a set of all image file names (without extensions) in the image directory
    image_files = {os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))}
    
    # Get a set of all xml file names (without extensions) in the xml directory
    xml_files = {os.path.splitext(f)[0] for f in os.listdir(xml_dir) if f.lower().endswith('.xml')}
    
    # Delete unmatched XML files
    for xml_file in xml_files:
        if xml_file not in image_files:
            xml_file_path = os.path.join(xml_dir, f"{xml_file}.xml")
            os.remove(xml_file_path)
            print(f"Deleted XML file: {xml_file_path}")
    
    # Delete unmatched image files
    for image_file in image_files:
        if image_file not in xml_files:
            image_file_path = os.path.join(image_dir, f"{image_file}.png")  # Adjust extension as necessary
            if os.path.exists(image_file_path):
                os.remove(image_file_path)
                print(f"Deleted image file: {image_file_path}")
            else:
                # Check for other common image formats
                for ext in ['.jpg', '.jpeg', '.bmp', '.tiff']:
                    potential_path = os.path.join(image_dir, f"{image_file}{ext}")
                    if os.path.exists(potential_path):
                        os.remove(potential_path)
                        print(f"Deleted image file: {potential_path}")
                        break

# Example usage:
image_directory = "Images_sorted"
xml_directory = "Annotations_XML"



match_and_clean_pairs(image_directory, xml_directory)
