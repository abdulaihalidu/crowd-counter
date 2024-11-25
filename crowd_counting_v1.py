import cv2
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os
import argparse

def load_annotations(annotation_path):
    df = pd.read_csv(annotation_path)
    columns = ["label", "x", "y", "image_name", "width", "height"]
    df.columns = columns
    return df

def load_image(image_path):
    # Load image in RGB format
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    return image

def draw_annotations_on_image(image, df):
    image_with_annotations = image.copy()
    for _, row in df.iterrows():
        x, y = row["x"], row["y"]
        cv2.circle(image_with_annotations, (x, y), 10, (0, 0, 255), -1)
    return image_with_annotations

def display_images_in_grid(image_titles_list, rows = None, cols = None, figsize=(12, 8)):
    if rows is None and cols is None:
        raise ValueError("Either rows or cols must be defined")
    # If rows is defined, calculate the number of columns
    if rows is not None:
        cols = int(np.ceil(len(image_titles_list) / rows))
    # If cols is defined, calculate the number of rows
    elif cols is not None:
        rows = int(np.ceil(len(image_titles_list) / cols))
    _, axes = plt.subplots(rows, cols, figsize=figsize)
    for i, ax in enumerate(axes.flat):
        if i < len(image_titles_list):
            if len(image_titles_list[i]) == 3:
                image, title, cmap = image_titles_list[i]
            else:
                image, title = image_titles_list[i]
                cmap = None
            ax.imshow(image, cmap=cmap)
            # if cmap is not None:
            # else:
            #     ax.imshow(image)
            ax.set_title(title)
            ax.axis("off")
        else:
            ax.axis("off")
    plt.tight_layout()
    plt.show()

def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def apply_errosion(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)

def apply_dilation(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

def apply_edge_detection(image, threshold1=100, threshold2=200):
    return cv2.Canny(image, threshold1, threshold2)

def apply_blob_detection(image):
    num_labels, labels_im = cv2.connectedComponents(image)
    return num_labels, labels_im

def apply_contour_detection(image):
    image_copy = image.copy()
    contours, _ = cv2.findContours(image_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 10)
    return image_copy, contours

def apply_binary_thresholding(image, threshold=128, max_value=255):
    _, thresholded_image = cv2.threshold(image, threshold, max_value, cv2.THRESH_BINARY)
    return thresholded_image

def apply_difference(image1, image2):
    return cv2.absdiff(image1, image2)

def crop_image_from_top_bottom(image, top_percentage=0, bottom_percentage=0):
    height, width = image.shape[:2]
    top = int(height * top_percentage)
    bottom = int(height * bottom_percentage)
    return image[top:height-bottom, :]

def apply_shadow_masking(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # Extract Saturation and Value channels
    _, _, value_channel = cv2.split(hsv_image)
    # Threshold based on brightness to filter shadows
    _, shadow_mask = cv2.threshold(value_channel, 100, 255, cv2.THRESH_BINARY)
    return shadow_mask

def apply_denoising(image):
    return cv2.fastNlMeansDenoising(image, None, 30, 7, 21)

def method_1(image):
    # Apply Denoising
    denoised_image = apply_denoising(image)
    # Apply Edge Detection
    edge_image = apply_edge_detection(denoised_image)
    # Apply Dilation
    dilated_image = apply_dilation(edge_image)
    # Final Image
    final_image = dilated_image
    # Apply contour detection
    final_image, contours = apply_contour_detection(final_image)
    num_labels = len(contours)
    return final_image, num_labels, contours

def method_2(image):
    # Apply Dilation
    dilated_image = apply_dilation(image)
    # Final Image
    final_image = dilated_image.copy()

    # Apply Connected Components
    num_labels, labels_im = apply_blob_detection(final_image)

    # Filter blobs based on size to detect people
    min_blob_area = 25  # Minimum area for a blob to be considered a person
    people_count = sum(
        cv2.countNonZero((labels_im == label).astype("uint8") * 255) > min_blob_area
        for label in range(1, num_labels)
    )
    return final_image, people_count, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--background", "-b", type=str, default="./Data/background.jpg", help="Path to the background image")
    parser.add_argument("--imgs_dir", "-i", type=str, default="./Data/Images", help="Path to the folder containing people images")
    parser.add_argument("--annotation", "-a", type=str, default="./Data/Annotations/labels.csv", help="Path to the annotation CSV file")
    args = parser.parse_args()

    # Load background image
    background_img = load_image(args.background)
    # Convert background image to grayscale
    background_gray = convert_to_gray(background_img)
    # Crop the top and bottom of the background image
    background_gray = crop_image_from_top_bottom(background_gray, 0.5, 0.1)
    # Load annotations
    annotations_df = load_annotations(args.annotation)
    # List to store images and their titles
    img_title_tuples = []
    
    # Loop through images in the directory
    list_of_images = os.listdir(args.imgs_dir)
    # list_of_images = ["1660647600.jpg", "1660662000.jpg", "1660644000.jpg", "1660636800.jpg", "1660633200.jpg"]
    # list_of_images = ["1660647600.jpg", "1660636800.jpg", "1660633200.jpg"]
    # list_of_images = ["1660647600.jpg", "1660633200.jpg"]
    # list_of_images = ["1660647600.jpg"]
    
    for img_name in list_of_images:
        img_path = os.path.join(args.imgs_dir, img_name)
        original_image = load_image(img_path)
        original_crowd_count = len(annotations_df[annotations_df["image_name"] == img_name])

        # Original Image
        original_image_with_annotations = draw_annotations_on_image(original_image, annotations_df[annotations_df["image_name"] == img_name])

        # Original Image Cropped
        original_image = crop_image_from_top_bottom(original_image, 0.5, 0.1)
        original_image_with_annotations = crop_image_from_top_bottom(original_image_with_annotations, 0.5, 0.1)
        original_image_title = f"Original Image: {img_name}, Number of people: {original_crowd_count}"
        img_title_tuples.append((original_image_with_annotations, original_image_title))
        
        # Convert image to grayscale
        gray_image = convert_to_gray(original_image)
        gray_image_title = "Grayscale Image"

        # Apply Difference
        difference_image = apply_difference(gray_image, background_gray)
        difference_image_title = "Difference Image"

        # Apply Threshold on Difference Image
        difference_thresholded_image = apply_binary_thresholding(difference_image, threshold=70)
        difference_thresholded_image_title = "Threshold on Difference Image"
        img_title_tuples.append((difference_thresholded_image, difference_thresholded_image_title, 'gray'))

        # Apply Shadow Masking
        shadow_mask = apply_shadow_masking(original_image)
        difference_thresholded_image = cv2.bitwise_and(difference_thresholded_image, difference_thresholded_image, mask=cv2.bitwise_not(shadow_mask))
        difference_thresholded_image_title = "Shadow Masking"
        img_title_tuples.append((difference_thresholded_image, "Shadow Masking", 'gray'))

        final_image, people_count, contours = method_1(difference_thresholded_image)
        final_image = original_image.copy()
        if contours is not None:
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(final_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        final_image_title = f"Number of People: {people_count}"
        img_title_tuples.append((final_image, final_image_title))
    
    # Display images in a grid
    display_images_in_grid(
        img_title_tuples,
        rows=None, 
        cols=2, 
        figsize=(12, 8)
    )