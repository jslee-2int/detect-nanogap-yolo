import json
import os
import uuid


def yolo_to_label_studio(yolo_file_path, image_width, image_height, class_map):
    """
    Convert YOLO label file to Label Studio JSON format.

    Args:
    - yolo_file_path (str): Path to the YOLO label file.
    - image_width (int): Width of the image.
    - image_height (int): Height of the image.
    - class_map (dict): Mapping of class_id to class name.

    Returns:
    - dict: Label Studio JSON data.
    """
    label_studio_data = {"annotations": [{"result": []}]}
    with open(yolo_file_path, "r") as file:
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.split())

            x_center *= image_width
            y_center *= image_height
            width *= image_width
            height *= image_height

            x = (x_center - width / 2) / image_width * 100
            y = (y_center - height / 2) / image_height * 100
            width = width / image_width * 100
            height = height / image_height * 100

            label = class_map[int(class_id)]

            label_studio_data["annotations"][0]["result"].append({
                "value": {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "rotation": 0,
                    "rectanglelabels": [label]
                },
                "id": str(uuid.uuid4()),
                "type": "rectanglelabels",
                "to_name": "image",
                "from_name": "label",
                "image_rotation": 0
            })
    return label_studio_data


# Example usage
yolo_file_path = "runs/detect/predict/labels/image2.txt"
image_width = 1100  # Replace with actual image width
image_height = 825  # Replace with actual image height
class_map = {0: "class_1", 1: "class_2"}  # Replace with actual class mapping

output_json = yolo_to_label_studio(yolo_file_path, image_width, image_height, class_map)

# Save JSON to file
output_file_path = "output.json"
with open(output_file_path, "w") as f:
    json.dump(output_json, f, indent=4)
