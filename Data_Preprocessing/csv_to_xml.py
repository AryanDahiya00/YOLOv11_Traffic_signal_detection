import os
import csv
import xml.etree.ElementTree as ET
from xml.dom import minidom

def create_xml(filename, width, height, objects, output_xml_path):
    # Create the root element
    annotation = ET.Element("annotation")
    
    # Filename element
    file_elem = ET.SubElement(annotation, "filename")
    file_elem.text = filename
    
    # Size element
    size = ET.SubElement(annotation, "size")
    width_elem = ET.SubElement(size, "width")
    width_elem.text = str(width)
    height_elem = ET.SubElement(size, "height")
    height_elem.text = str(height)  # Ensure height is set correctly
    depth_elem = ET.SubElement(size, "depth")
    depth_elem.text = "3"  # Assuming RGB images
    
    # Add all objects (bounding boxes and classes)
    for obj_class, xmin, ymin, xmax, ymax in objects:
        obj = ET.SubElement(annotation, "object")
        name_elem = ET.SubElement(obj, "name")
        name_elem.text = obj_class
        bndbox = ET.SubElement(obj, "bndbox")
        
        xmin_elem = ET.SubElement(bndbox, "xmin")
        xmin_elem.text = str(xmin)
        ymin_elem = ET.SubElement(bndbox, "ymin")
        ymin_elem.text = str(ymin)
        xmax_elem = ET.SubElement(bndbox, "xmax")
        xmax_elem.text = str(xmax)
        ymax_elem = ET.SubElement(bndbox, "ymax")
        ymax_elem.text = str(ymax)
    
    # Convert to string with pretty print
    xml_str = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="   ")
    
    # Write to XML file
    with open(output_xml_path, "w") as f:
        f.write(xml_str)

def process_csv(csv_file, output_base_dir, image_width, image_height):
    annotations = {}

    with open(csv_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        next(reader)  # Skip the header if there is one
        
        # Process each row in the CSV file
        for row in reader:
            filename = row[0]  # Filename
            obj_class = row[1]  # Annotation tag (class label)
            xmin = int(row[2])  # Upper left corner X
            ymin = int(row[3])  # Upper left corner Y
            xmax = int(row[4])  # Lower right corner X
            ymax = int(row[5])  # Lower right corner Y
            
            # Group annotations by filename to handle multiple objects in the same image
            if filename not in annotations:
                annotations[filename] = []
            
            annotations[filename].append((obj_class, xmin, ymin, xmax, ymax))
    
    # Create XML files for each image (filename)
    for filename, objects in annotations.items():
        xml_filename = f"{os.path.splitext(os.path.basename(filename))[0]}.xml"
        output_xml_path = os.path.join(output_base_dir, xml_filename)
        
        # Create XML with multiple objects for this image
        create_xml(os.path.basename(filename), image_width, image_height, objects, output_xml_path)

def convert_annotations(input_dir, output_dir, image_width, image_height):
    # Walk through the input directory recursively
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file == 'frameAnnotationsBULB.csv':
                csv_file_path = os.path.join(root, file)
                
                # Generate the corresponding output directory path, maintaining the same structure
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                
                # Ensure the output sub-directory exists
                os.makedirs(output_subdir, exist_ok=True)
                
                # Process the CSV file and generate XML files
                process_csv(csv_file_path, output_subdir, image_width, image_height)

# Example usage:
input_dir = "Original/Annotations"  # This is the base directory with daytrain, nighttrain, etc.
output_dir = "Annotations_xml"  # This will hold the XML files with the same structure
image_width = 1280  # Replace with actual image width
image_height = 960  # Replace with actual image height

convert_annotations(input_dir, output_dir, image_width, image_height)
