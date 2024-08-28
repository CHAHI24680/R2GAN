import cv2
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Set up path to your input/output directory
input_dir = r'..\images_Cityscapes'
input_dir2 = r'..\maps_Cityscapes'
output_dir = r'..\PMF_train_Cityscapes'

# Get lists of files for each image type
real_a_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
real_b_files = [f for f in os.listdir(input_dir2) if f.endswith('.png')]

# Define function for image fusion
def image_fusion(real_a_path, real_b_path):
    # Load the two images
    real_a = cv2.imread(os.path.join(input_dir, real_a_path))
    real_a = cv2.resize(real_a, (256,256))

    real_b = cv2.imread(os.path.join(input_dir2, real_b_path))
    real_b = cv2.resize(real_b, (256,256))

    real_b_gray = cv2.imread(os.path.join(input_dir2, real_b_path), 0)
    real_b_gray = cv2.resize(real_b_gray, (256,256))


    # Load the RGB image and its corresponding segmentation map

    img = real_a
    seg_map_gry = real_b_gray
    seg_map = real_b


    # Print out the unique pixel values in the segmentation map
    unique_values = np.unique(seg_map.reshape(-1, seg_map.shape[2]), axis=0)
    #print(unique_values)

    # Convert the segmentation map to a binary mask
    mask = np.zeros_like(seg_map)
    mask2 = np.zeros_like(seg_map)
    #mask[np.where(np.all(seg_map == [142, 0, 0], axis=-1))] = 255 # apply blur to objects labeled as Red

    # Blur only the first half of the segments
    num_segments = len(unique_values) // 2
    for i, value in enumerate(unique_values):
        if i < num_segments:
            # Create a binary mask for the current segment
            #mask = np.where(seg_map_gray == value, 255, 0).astype(np.uint8)
            mask[np.where(np.all(seg_map == value, axis=-1))] = 255
        else:
            mask2[np.where(np.all(seg_map == value, axis=-1))] = 255
        

    # Apply a Gaussian blur filter to the masked regions of the image

    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    gray_mask2 = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img, (35, 35), 0)
    blur_masked = cv2.bitwise_and(blur, blur, mask=gray_mask)
    blur_masked2 = cv2.bitwise_and(blur, blur, mask=gray_mask2)


    # Combine the blurred and sharp regions of the image using the mask

    sharp_masked = cv2.bitwise_and(img, img, mask=np.bitwise_not(gray_mask))
    sharp_masked2 = cv2.bitwise_and(img, img, mask=np.bitwise_not(gray_mask2))
    result = cv2.addWeighted(sharp_masked, 1.0, blur_masked, 1.0, 0.0)
    result2 = cv2.addWeighted(sharp_masked2, 1.0, blur_masked2, 1.0, 0.0)


    # Save the deblurred image to a file
    #cv2.imwrite('deblurred_image1.jpg', result)
    #cv2.imwrite('deblurred_image2.jpg', result2)

    stacked_image = np.hstack((result, result2, img))

    # Construct the output filename and write the stacked image
    stacked_filename = real_a_path.replace('.png', '_stacked.png')
    cv2.imwrite(os.path.join(output_dir, stacked_filename), stacked_image)

# Use ThreadPoolExecutor to parallelize the image fusion across multiple threads
with ThreadPoolExecutor(max_workers=20) as executor:
    executor.map(image_fusion, real_a_files, real_b_files)
