import cv2
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os
import argparse
import logging
from typing import Optional

class CrowdCounter:
    def __init__(self, 
                 background_img_path: str, 
                 annotation_path: str, 
                 people_images_folder: str, 
                 output_dir: str, 
                 crop_top_percentage: float = 0.5, 
                 crop_bottom_percentage: float = 0.1,
                 erosion_kernel_size: tuple[int, int] = (5, 5),
                 dilation_kernel_size: tuple[int, int] = (5, 5),
                 canny_threshold1: int = 100, 
                 canny_threshold2: int = 200,
                 binary_threshold: int = 70,
                 shadow_threshold: int = 100,
                 min_blob_area: int = 25,
                 bbox_overlap_threshold: float = 0.8,
                 point_in_bbox_threshold: int = 20):
        
        # Input validation
        if not os.path.exists(background_img_path):
            raise FileNotFoundError(f"Background image not found: {background_img_path}")
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
        if not os.path.isdir(people_images_folder):
            raise NotADirectoryError(f"People images directory not found: {people_images_folder}")

        # Configure logging
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)

        # Store configuration parameters
        self.background_img_path = background_img_path
        self.people_images_folder = people_images_folder
        self.csv_file_path = annotation_path
        self.crop_top_percentage = crop_top_percentage
        self.crop_bottom_percentage = crop_bottom_percentage
        self.output_dir = output_dir

        # Processing parameters
        self.erosion_kernel_size = erosion_kernel_size
        self.dilation_kernel_size = dilation_kernel_size
        self.canny_threshold1 = canny_threshold1
        self.canny_threshold2 = canny_threshold2
        self.binary_threshold = binary_threshold
        self.shadow_threshold = shadow_threshold
        self.min_blob_area = min_blob_area
        
        # Evaluation parameters
        self.bbox_overlap_threshold = bbox_overlap_threshold
        self.point_in_bbox_threshold = point_in_bbox_threshold
        
        # Metrics tracking
        self.total_images = 0
        self.total_ground_truth_points = 0
        self.correctly_detected_points = 0

        # Create output directory
        self._make_output_dir(output_dir)
        
        # Load annotations and preprocess background
        self.df = self._load_annotation()
        self.background_gray = self._load_and_preprocess_background_img()

    def _make_output_dir(self, output_dir: str) -> None:
        """Create output directory if it doesn't exist."""
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Output directory created: {output_dir}")

    def _load_and_preprocess_background_img(self) -> np.ndarray:
        """Load and preprocess background image."""
        background_img = self._load_image(self.background_img_path)
        background_gray = self._convert_to_gray(background_img)
        background_gray = self._crop_image(background_gray, 
                                           top_percentage=self.crop_top_percentage, 
                                           bottom_percentage=self.crop_bottom_percentage)
        return background_gray
        
    def _load_annotation(self) -> pd.DataFrame:
        """Load annotations from CSV file."""
        columns = ["label", "x", "y", "image_name", "width", "height"]
        try:
            df = pd.read_csv(self.csv_file_path, header=None, names=columns)
            self.logger.info(f"Loaded annotations: {len(df)} entries")
            return df
        except Exception as e:
            self.logger.error(f"Error loading annotations: {e}")
            raise

    def _load_image(self, image_path: str) -> np.ndarray:
        """Load an image with error handling."""
        try:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image is None:
                raise FileNotFoundError(f"Image not found at path: {image_path}")
            return image
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {e}")
            raise

    def _draw_annotations_on_image(self, image: np.ndarray, image_df_subset: pd.DataFrame) -> np.ndarray:
        """Draw point annotations on image."""
        image_with_annotations = image.copy()
        for _, row in image_df_subset.iterrows():
            x, y = row["x"], row["y"]
            cv2.circle(image_with_annotations, (x, y), 5, (255, 165, 0), -1)
        return image_with_annotations

    def _convert_to_gray(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale."""
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def _apply_erosion(self, image: np.ndarray) -> np.ndarray:
        """Apply erosion to image."""
        kernel = np.ones(self.erosion_kernel_size, np.uint8)
        return cv2.erode(image, kernel, iterations=1)

    def _apply_dilation(self, image: np.ndarray) -> np.ndarray:
        """Apply dilation to image."""
        kernel = np.ones(self.dilation_kernel_size, np.uint8)
        return cv2.dilate(image, kernel, iterations=1)

    def _apply_canny_edge_detector(self, image: np.ndarray) -> np.ndarray:
        """Apply Canny edge detection."""
        return cv2.Canny(image, self.canny_threshold1, self.canny_threshold2)


    def _detect_contours(self, image: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """Detect contours in image."""
        image_copy = image.copy()
        contours, _ = cv2.findContours(image_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 10)
        return image_copy, contours

    def _apply_binary_thresholding(self, image: np.ndarray, threshold: Optional[int] = None) -> np.ndarray:
        """Apply binary thresholding."""
        threshold = threshold or self.binary_threshold
        _, thresholded_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        return thresholded_image

    def _apply_difference(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        """Calculate absolute difference between two images."""
        return cv2.absdiff(image1, image2)

    def _crop_image(self, 
                    image: np.ndarray, 
                    top_percentage: float = 0, 
                    bottom_percentage: float = 0) -> np.ndarray:
        """
        Crop an image from top and bottom based on percentage.
        
        Args:
            image (np.ndarray): Input image
            top_percentage (float): Percentage to crop from top
            bottom_percentage (float): Percentage to crop from bottom
        
        Returns:
            np.ndarray: Cropped image
        """
        height, width = image.shape[:2]
        top = int(height * top_percentage)
        bottom = int(height * bottom_percentage)
        return image[top:height-bottom, :]

    def _apply_shadow_masking(self, image: np.ndarray) -> np.ndarray:
        """Apply shadow masking to image."""
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        _, _, value_channel = cv2.split(hsv_image)
        _, shadow_mask = cv2.threshold(value_channel, self.shadow_threshold, 255, cv2.THRESH_BINARY)
        return shadow_mask

    def _apply_denoising(self, image: np.ndarray) -> np.ndarray:
        """Apply non-local means denoising."""
        return cv2.fastNlMeansDenoising(image, None, 30, 7, 21)
    
    
    def _is_bbox_inside_other(self, bbox1: np.ndarray, bbox2: np.ndarray) -> bool:
        """
        Check if one bounding box is mostly inside another.
        
        Args:
            bbox1 (np.ndarray): First bounding box [x, y, w, h]
            bbox2 (np.ndarray): Second bounding box [x, y, w, h]
        
        Returns:
            bool: True if bbox1 is mostly inside bbox2, False otherwise
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection area
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return False
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bbox1_area = w1 * h1
        
        # Check if intersection is a significant portion of bbox1
        return intersection_area / bbox1_area > self.bbox_overlap_threshold

    def _filter_nested_bounding_boxes(self, contours: list[np.ndarray]) -> list[np.ndarray]:
        """
        Remove bounding boxes that are mostly inside other bounding boxes.
        
        Args:
            contours (List[np.ndarray]): List of contours
        
        Returns:
            List[np.ndarray]: Filtered list of contours
        """
        # Get bounding rectangles
        bboxes = [cv2.boundingRect(contour) for contour in contours]
        
        # Filter out nested bounding boxes
        filtered_bboxes = []
        for i, bbox1 in enumerate(bboxes):
            is_nested = any(
                self._is_bbox_inside_other(bbox1, bbox2) 
                for j, bbox2 in enumerate(bboxes) if i != j
            )
            if not is_nested:
                filtered_bboxes.append(bboxes[i])
        
        return filtered_bboxes

    def _point_in_bbox(self, point: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
        """
        Check if a point is within or close to a bounding box.
        
        Args:
            point (Tuple[int, int]): (x, y) coordinates of the point
            bbox (Tuple[int, int, int, int]): Bounding box (x, y, width, height)
        
        Returns:
            bool: True if point is within or close to bbox, False otherwise
        """
        x, y = point
        bx, by, bw, bh = bbox
        
        # Check if point is within bbox
        in_x = bx - self.point_in_bbox_threshold <= x <= bx + bw + self.point_in_bbox_threshold
        in_y = by - self.point_in_bbox_threshold <= y <= by + bh + self.point_in_bbox_threshold
        
        return in_x and in_y

    def _calculate_correct_detections(self, 
                            img_df: pd.DataFrame, 
                            detected_bboxes: list[tuple[int, int, int, int]]) -> tuple[int, int]:
        """
        Calculate the number of correctly detected bounding boxes
        
        Args:
            img_df (pd.DataFrame): DataFrame with ground truth point annotations
            detected_bboxes (List[Tuple[int, int, int, int]]): List of detected bounding boxes
        
        Returns:
            List[int]: List of indices of correctly detected bounding boxes
        """
        # Initialize list to store correct detection indices
        detections = []
        
        for idx, row in img_df.iterrows():
            point = (row['x'], row['y'])
            closest_bbxs = []
            for bbox_idx, bbox in enumerate(detected_bboxes):
                if self._point_in_bbox(point, bbox):
                    closest_bbxs.append((bbox_idx, bbox))
            if closest_bbxs:
                if len(closest_bbxs) == 1:
                    detections.append(closest_bbxs[0][0])
                else:
                    closest_bbxs_distances = [np.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2) for idx, (x, y, _, _) in closest_bbxs]
                    detections.append(closest_bbxs[np.argmin(closest_bbxs_distances)][0]) 
        return detections
    
    def _mean_root_squared_error(self,num_people_truth, num_people_detected):
        """
        Calculate the Mean Root Squared Error between actual and detected number of people.
        
        Args:
            num_people_truth (list or np.ndarray): Ground truth number of people for each image.
            num_people_detected (list or np.ndarray): Detected number of people for each image.
        
        Returns:
            float: Mean Squared Error.
        """
        # Convert lists to numpy arrays
        num_people_truth = np.array(num_people_truth)
        num_people_detected = np.array(num_people_detected)
        
        squared_diffs = (num_people_truth - num_people_detected) ** 2

        mrse = np.sqrt(np.mean(squared_diffs))
        
        return mrse
    
    def _save_result_image(self, 
                            original_img: np.ndarray, 
                            binary_thresh_img: np.ndarray, 
                            shadow_mask_img: np.ndarray, 
                            final_img: np.ndarray, 
                            output_path: str, 
                            original_img_people_count: int, 
                            detected_people_count: int) -> None:
        """Save result images with visualization."""
        plt.figure(figsize=(20, 12), dpi=600)
        
        plot_titles = [
            f"Original Image (Ground Truth: {original_img_people_count})",
            "Binary Thresholding Result",
            "Shadow Mask",
            f"Detected People: {detected_people_count}"
        ]
        
        for i, (img, title) in enumerate(
            zip([original_img, binary_thresh_img, shadow_mask_img, final_img], plot_titles), 1
        ):
            plt.subplot(2, 2, i)
            plt.imshow(img if i != 1 else img, cmap="gray" if i in [2, 3] else None)
            plt.title(title)
        
        plt.tight_layout()
        plt.savefig(output_path)
        self.logger.info(f"Saved result image: {output_path}")

    def _detect_crowd(self, image: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """Method for crowd detection."""
        denoised_image = self._apply_denoising(image)
        edge_image = self._apply_canny_edge_detector(denoised_image)
        dilated_image = self._apply_dilation(edge_image)
        final_image = dilated_image.copy()
        final_image, contours = self._detect_contours(final_image)

        return final_image, contours

    def run(self) -> None:
        """
        Enhanced run method to filter bounding boxes and track accuracy.
        """
        # Reset metrics
        self.total_images = 0
        self.total_ground_truth_points = 0
        self.correctly_detected_points = 0 

        num_people_truth = []
        num_people_detected = []    

        self.logger.info("Starting crowd counting...")

        for filename in os.listdir(self.people_images_folder):
            if filename.lower().endswith(".jpg"):
                image_path = os.path.join(self.people_images_folder, filename)
                img_df = self.df[self.df["image_name"] == f"{filename.split('.')[0]}.jpg"]
                try:
                    original_image = self._load_image(image_path)
                    
                    original_crowd_count = len(img_df)

                    original_image_with_gt_annotation = self._draw_annotations_on_image(
                        original_image, img_df)
                    
                    cropped_original_img = self._crop_image(
                        original_image, 
                        self.crop_top_percentage, 
                        self.crop_bottom_percentage
                    )
                    gray_image = self._convert_to_gray(cropped_original_img)

                    # Apply Difference
                    difference_image = self._apply_difference(gray_image, self.background_gray)
                    
                    # Apply Threshold on Difference Image
                    difference_thresholded_image = self._apply_binary_thresholding(difference_image)
                    
                    # Apply Shadow Masking
                    shadow_mask = self._apply_shadow_masking(cropped_original_img)
                    difference_thresholded_image = cv2.bitwise_and(
                        difference_thresholded_image, difference_thresholded_image, 
                        mask=cv2.bitwise_not(shadow_mask)
                    )
                    
                    _, contours = self._detect_crowd(difference_thresholded_image)
                    
                    original_image_with_all_annotations = original_image_with_gt_annotation.copy()
                    # Filter nested bounding boxes
                    if contours is not None:
                        original_img = original_image.copy()
                        # Calculate top offset for drawing boxes
                        top = int(original_image.shape[0] * self.crop_top_percentage)
                        
                        # Get filtered bounding boxes
                        filtered_bboxes = self._filter_nested_bounding_boxes(contours)
                        
                        # Calculate accuracy
                        correct_detections = self._calculate_correct_detections(
                            img_df, 
                            [(x, y+top, w, h) for (x, y, w, h) in filtered_bboxes])
                        
                        # Draw filtered bounding boxes
                        for bbox_idx, bbox in enumerate(filtered_bboxes):
                            # Default color is red
                            color = (255, 0, 0)
                            # If bbox is in correct detections (which has the index of the point and bbox)
                            if bbox_idx in correct_detections:
                                color = (0, 255, 0)
                            x, y, w, h = bbox
                            cv2.rectangle(original_image_with_all_annotations, (x, y+top), (x+w, y+h+top), color, 2)
                        
                        # Update overall metrics
                        self.total_images += 1
                        self.total_ground_truth_points += len(img_df)
                        self.correctly_detected_points += len(correct_detections)

                        num_people_truth.append(original_crowd_count)
                        num_people_detected.append(len(filtered_bboxes))
                    
                    # Save result image
                    self._save_result_image(
                        original_image_with_gt_annotation, 
                        difference_thresholded_image, 
                        shadow_mask,
                        original_image_with_all_annotations, 
                        f"{self.output_dir}/{filename.split('.')[0]}.png", 
                        original_crowd_count, 
                        len(filtered_bboxes) if contours is not None else 0
                    )
                except FileNotFoundError as e:
                    self.logger.error(f"File not found: {filename} - {e}")
                except Exception as e:
                    self.logger.error(f"Error processing {filename}: {e}")
        
        # Print overall accuracy
        if self.total_images > 0:
            precision = self.correctly_detected_points / sum(num_people_detected)
            recall = self.correctly_detected_points / sum(num_people_truth)
            mrse = self._mean_root_squared_error(num_people_truth, num_people_detected)
            print("")
            print("*" * 76)
            self.logger.info(f"Total Images Processed: {self.total_images}")
            self.logger.info(f"Mean Root Squared Error (MRSE): {mrse:.2f}")
            self.logger.info(f"Precision: {precision * 100:.2f}:%")
            self.logger.info(f"Recall: {recall * 100:.2f}:%")
            print("*" * 76)
            self.logger.info(f"Total Ground Truth Points: {self.total_ground_truth_points}")
            self.logger.info(f"Total Detected Points: {sum(num_people_detected)}")
            self.logger.info(f"Correctly Detected Points: {self.correctly_detected_points}")
            print("*" * 76)
        self.logger.info("Crowd counting completed!")

def main():
    """Main function to parse arguments and run crowd counter."""
    parser = argparse.ArgumentParser(description="Crowd Counting Computer Vision Tool")
    parser.add_argument("--background", "-b", type=str, 
                        default="./Data/background.jpg", 
                        help="Path to the background image")
    parser.add_argument("--imgs_dir", "-i", type=str, 
                        default="./Data/Images", 
                        help="Path to the folder containing people images")
    parser.add_argument("--annotation", "-a", type=str, 
                        default="./Data/Annotations/labels.csv", 
                        help="Path to the annotation CSV file")
    parser.add_argument("--output_dir", "-o", type=str, 
                        default="./Results", 
                        help="Path to the output directory")
    parser.add_argument("--crop_top_percentage", "-ct", type=float, 
                        default=0.5, 
                        help="Percentage of the image to crop from the top")    
    parser.add_argument("--crop_bottom_percentage", "-cb", type=float, 
                        default=0.1,    
                        help="Percentage of the image to crop from the bottom")
    parser.add_argument("--bbox_overlap_threshold", "-bot", type=float, 
                        default=0.8,    
                        help="Threshold for considering one bounding box inside another")
    parser.add_argument("--point_in_bbox_threshold", "-pit", type=int, 
                        default=20,    
                        help="Pixel distance threshold for considering a point in a bounding box")
    
    args = parser.parse_args()

    crowd_counter = CrowdCounter(
        args.background, 
        args.annotation, 
        args.imgs_dir, 
        args.output_dir, 
        args.crop_top_percentage, 
        args.crop_bottom_percentage,
        bbox_overlap_threshold=args.bbox_overlap_threshold,
        point_in_bbox_threshold=args.point_in_bbox_threshold
    )
    crowd_counter.run() 

if __name__ == "__main__":
    main()