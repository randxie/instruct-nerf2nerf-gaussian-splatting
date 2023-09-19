import os
from PIL import Image

def downsize_image(input_path, output_path, target_width):
    # Open the original image
    img = Image.open(input_path)
    
    # Calculate the aspect ratio
    aspect_ratio = img.height / img.width
    
    # Calculate the new height maintaining the aspect ratio
    new_height = int(target_width * aspect_ratio)
    
    # Resize the image
    img_resized = img.resize((target_width, new_height))
    
    # Save the resized image to the output path
    img_resized.save(output_path)

def process_images_in_directory(input_dir, output_dir, target_width=512):
    # Ensure output directory exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List all files in the input directory
    for filename in os.listdir(input_dir):
        # Build full file path
        file_path = os.path.join(input_dir, filename)
        
        # Check if it's an image based on its file extension (you can add more if needed)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            # Build the output file path
            output_path = os.path.join(output_dir, filename)
            
            # Downsize the image
            downsize_image(file_path, output_path, target_width)

# Example usage:
input_directory = "data/bear_cm/images_ori"
output_directory = "data/bear_cm/images"

os.makedirs(output_directory, exist_ok=True)
process_images_in_directory(input_directory, output_directory)

