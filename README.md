Crowd-Counting Algorithm based on pixel-level operations

## Overview
This Python-based Crowd Counting Computer Vision Tool is designed to detect and count people in images using  image processing techniques. The tool leverages OpenCV and other image processing libraries to analyze crowd density and provide an estimate of the number of  people in a given image.

## Features
#### Advanced Image Processing: Utilizes multiple image processing techniques including:
- Canny edge detection
- Binary thresholding
- Shadow masking
- Denoising

#### Adaptive Cropping: Configurable top and bottom image cropping
#### Bounding Box Filtering: Removes nested or redundant bounding boxes
#### Accuracy Metrics: Provides comprehensive performance metrics including:
- Mean Root Squared Error (MRSE)
- Precision
- Recall



## Prerequisites

Python 3.8+

### Required Libraries:
- OpenCV (cv2)
- NumPy
- Pandas
- Matplotlib



## Installation

Clone the repository:
>git clone https://github.com/abdulaihalidu/crowd-counter.git
>cd crowd-counter

Install required dependencies:

>install -r requirements.txt

### Usage
>bashCopypython crowd_counter.py [OPTIONS]
Options:

--background, -b: Path to background image (default: ./Data/background.jpg)
--imgs_dir, -i: Directory containing people images (default: ./Data/Images)
--annotation, -a: Path to annotation CSV file (default: ./Data/Annotations/labels.csv)
--output_dir, -o: Output directory for results (default: ./Results)
--crop_top_percentage, -ct: Percentage to crop from top (default: 0.5)
--crop_bottom_percentage, -cb: Percentage to crop from bottom (default: 0.1)
--bbox_overlap_threshold, -bot: Bounding box overlap threshold (default: 0.8)
--point_in_bbox_threshold, -pit: Point-in-bbox pixel threshold (default: 20)

Example
>python crowd_counter.py -b ./background.jpg -i ./test_images -o ./results -ct 0.4 -cb 0.2
Output
The tool generates:

Detailed result images in the specified output directory
Logging information in console
Performance metrics including:

Total images processed
Mean Root Squared Error
Precision and Recall
Total ground truth and detected points

### Algorithm Overview

#### Image Preprocessing
- Load background and test images
- Convert to grayscale
- Apply cropping
- Denoise images

#### Difference Detection
- Calculate image difference from background
- Apply binary thresholding
- Perform shadow masking


#### Crowd Detection
- Apply Canny edge detection
- Detect and filter contours
- Remove nested bounding boxes

#### Performance Evaluation
- Match detected points with ground truth
- Calculate accuracy metrics



## Customization
The CrowdCounter class allows extensive customization through initialization parameters, enabling fine-tuning for different scenarios.

### Limitations
- Requires a static background image
- Performance depends on image quality and complexity
- Best suited for controlled environments

## Contributing
Contributions are welcome! Please submit pull requests or open issues to suggest improvements or report bugs.
License

Free to use provided credits is given

### Authors
- Halidu Abdulai
- Ahmed Kamal Baig
