import os
import xml.etree.ElementTree as ET

def remove_specific_objects_from_xml(source_folder, objects_to_remove):
    # Walk through all files in the source folder
    for filename in os.listdir(source_folder):
        if filename.endswith('.xml'):
            file_path = os.path.join(source_folder, filename)
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Find and remove the specified objects
            for obj in root.findall('object'):
                name_element = obj.find('name')
                if name_element is not None and name_element.text in objects_to_remove:
                    root.remove(obj)  # Remove the object if it matches
            
            # Save the modified XML back to the same file
            tree.write(file_path)
            print(f'Updated: {file_path}')

# Example usage
source_folder = 'Annotations_XML'  # Replace with the path to your XML files
objects_to_remove = ['goLeft','stopLeft','warningLeft','goForward']  # Objects to remove

remove_specific_objects_from_xml(source_folder, objects_to_remove)
