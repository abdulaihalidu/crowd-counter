import cv2
import numpy as np
import pandas as pd 
from skimage.exposure import match_histograms
import matplotlib.pyplot as plt
import os
import argparse

class CrowdCounter:
    def __init__(self, background_img_path, annotation_path,people_images_folder):
        self.background_img_path = background_img_path
        self.people_images_folder = people_images_folder
        self.csv_file_path = annotation_path
        self.image_gray = None
        self.image_diff = None
        self.image_thres = None
        self.df = self._load_process_annotation()
        # Define CLAHE and morphological kernel for image processing
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # Load and preprocess the background image
        self.background_img, self.background_gray = self._load_and_preprocess_background()

    def _load_process_annotation(self):
        df = pd.read_csv(self.csv_file_path)
        columns = ["label", "x", "y", "image_name", "width", "height"]
        df.columns = columns

        return df

    def _load_and_preprocess_background(self):
        """Load and preprocess the background image."""
        background_img = cv2.imread(self.background_img_path, cv2.IMREAD_COLOR)
        if background_img is None:
            raise FileNotFoundError(f"Background image not found at path: {self.background_img_path}")

        background_gray = cv2.cvtColor(background_img, cv2.COLOR_BGR2GRAY)
        background_gray = self.clahe.apply(background_gray)

        return background_img, background_gray

    def process_image(self, image_path):
        """Process a single image file and count the number of people."""
        # Load and resize image to match the background size
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        if image.shape[:2] != self.background_img.shape[:2]:
            image = cv2.resize(image, (self.background_img.shape[1], self.background_img.shape[0]))

        # Convert image to grayscale and apply CLAHE
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gray = self.clahe.apply(image_gray)
        
        # Matching Histograms
        # image_gray = match_histograms(image_gray, self.background_gray)
        # Convert to Matlike Image from 
        # image_gray = np.clip(image_gray, 0, 255).astype(np.uint8)
        self.image_gray = image_gray

        # Background Subtraction
        difference = cv2.absdiff(self.background_gray, image_gray)
        _, foreground_mask = cv2.threshold(difference, 20, 255, cv2.THRESH_BINARY)
        self.image_diff = difference
        self.image_thres = foreground_mask

        # Applying Canny Edge Detector for edge detection
        edges = cv2.Canny(foreground_mask, 30, 150)

        # Applying Morphological Operations (Dilation and Erosion)
        dilated = cv2.dilate(edges, self.kernel, iterations=1)
        eroded = cv2.erode(dilated, self.kernel, iterations=1)

        # Blob Detection using connected components
        num_labels, labels_im = cv2.connectedComponents(eroded)

        # Filter blobs based on size to detect people
        min_blob_area = 150  # Minimum area for a blob to be considered a person
        people_count = sum(
            cv2.countNonZero((labels_im == label).astype("uint8") * 255) > min_blob_area
            for label in range(1, num_labels)
        )

        return people_count, foreground_mask, edges, eroded, labels_im

    def count_people(self):
        """Count people in all images within the specified directory."""
        results = []
        for filename in os.listdir(self.people_images_folder):
            if filename.lower().endswith(".jpg"):
                image_path = os.path.join(self.people_images_folder, filename)
                img_df = self.df[self.df["image_name"] == filename]
                try:
                    people_count, foreground_mask, edges, eroded, labels_im = self.process_image(image_path)
                    results.append((filename, people_count))

                    # Visualization for each image
                    plt.figure(figsize=(12, 8))
                    plt.suptitle(f'Processing: {filename} - People Count: {people_count}', fontsize=16)
                    img = cv2.imread(image_path)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # plt.subplot(321), plt.imshow(img_rgb)
                    # plt.title('Original Image')
                    img_copy = img_rgb.copy()
                    # draw circles df[['x', 'y'] with radius 5
                    for index, row in img_df.iterrows(): 
                        x, y = row['x'], row['y']
                        cv2.circle(img_copy, (x, y), 10, (0, 0, 255), -1)
                    plt.subplot(321), plt.imshow(img_copy)
                    plt.title(f'Original Image with {len(img_df)} people')
                    plt.subplot(322), plt.imshow(self.background_gray, cmap='gray')
                    plt.title(f'Contrast Background Image')
                    plt.subplot(323), plt.imshow(self.image_gray, cmap='gray')
                    plt.title(f'Contrast Original Image')
                    plt.subplot(324), plt.imshow(self.image_diff, cmap='gray')
                    plt.title(f'Contrast Difference Image')
                    plt.subplot(325), plt.imshow(self.image_thres, cmap='gray')
                    plt.title(f'Contrast Thresholded Image')
                    # plt.subplot(323), plt.imshow(foreground_mask, cmap='gray')
                    # # plt.title('Foreground Mask')
                    # plt.subplot(324), plt.imshow(edges, cmap='gray')
                    # plt.title('Edges Detected')
                    # plt.subplot(326), plt.imshow(eroded, cmap='gray')
                    # plt.title('Morphological Operations')
                    plt.subplot(326), plt.imshow(labels_im, cmap='gray')
                    plt.title('Connected Components')
                    plt.show()
                except FileNotFoundError as e:
                    print(e)

        # Print the count of people for each image
        for filename, count in results:
            print(f"Image: {filename}, Number of people detected: {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--background", "-b", type=str, required=True, help="Path to the background image")
    parser.add_argument("--imgs_dir", "-i", type=str, required=True, 
                        help="Path to the folder containing people images")
    parser.add_argument("--annotation", "-a", type=str, default="./Data/Annotations/labels.csv",
                        help="Path to the annotation CSV file")
    args = parser.parse_args()

    crowd_counter = CrowdCounter(args.background, args.annotation, args.imgs_dir)
    crowd_counter.count_people()
