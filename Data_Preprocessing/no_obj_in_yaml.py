import os
import xml.etree.ElementTree as ET

def delete_empty_xml_files(source_folder):
    # Walk through all files in the source folder
    for filename in os.listdir(source_folder):
        if filename.endswith('.xml'):
            file_path = os.path.join(source_folder, filename)
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()

                # Check for the presence of 'object' tags
                objects = root.findall('object')
                
                if not objects:  # If there are no objects, delete the file
                    os.remove(file_path)
                    print(f'Deleted: {file_path}')
            except ET.ParseError:
                print(f'Failed to parse: {file_path}. It may be corrupted.')

# Example usage
source_folder = 'Annotations_XML'  # Replace with the path to your XML files

delete_empty_xml_files(source_folder)
