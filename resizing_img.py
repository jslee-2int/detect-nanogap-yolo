from PIL import Image
import os

input_folder = 'raw_img'
output_folder = 'resized_img'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

new_size = (1100, 825)

for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)

    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
        try:
            with Image.open(file_path) as img:
                resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
                output_path = os.path.join(output_folder, filename)
                resized_img.save(output_path)
                print(f"Resized and saved: {output_path}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    else:
        print(f"Skipping non-image file: {filename}")

print("Resizing completed!")