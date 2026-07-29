from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Tuple, Union
import math


class ParticleDetector:
    def __init__(self, model_path: str,
                 grid_size: int = 4,
                 iou_threshold: float = 0.3,
                 padding: int = 50,
                 confidence_threshold: float = 0.25,
                 aspect_ratio_threshold: float = 0.7):  # 비율 임계값 추가
        """
        Initialize ParticleDetector with customizable parameters.

        Args:
            model_path: Path to the YOLO model weights
            grid_size: Number of grid divisions for processing
            iou_threshold: IoU threshold for merging boxes
            padding: Padding size for grid processing
            confidence_threshold: Minimum confidence score for detections
            aspect_ratio_threshold: Threshold for circle/rectangle decision (0-1)
        """
        self.model = YOLO(model_path)
        self.grid_size = grid_size
        self.iou_threshold = iou_threshold
        self.padding = padding
        self.confidence_threshold = confidence_threshold
        self.aspect_ratio_threshold = aspect_ratio_threshold

    def split_image(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Split image into grid with padding to handle boundary objects."""
        h, w, _ = image.shape
        h_step, w_step = h // self.grid_size, w // self.grid_size
        sub_images = []

        # print(min(h, w), self.padding)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Calculate base coordinates
                x1, y1 = j * w_step, i * h_step
                x2, y2 = x1 + w_step, y1 + h_step

                # Add padding with boundary checking
                pad_x1 = max(0, x1 - self.padding)
                pad_y1 = max(0, y1 - self.padding)
                pad_x2 = min(w, x2 + self.padding)
                pad_y2 = min(h, y2 + self.padding)

                # Extract padded sub-image
                sub_img = image[pad_y1:pad_y2, pad_x1:pad_x2]

                # Store with original coordinates and padding offsets
                sub_images.append((
                    sub_img,
                    (pad_x1, pad_y1, x1, y1, x2, y2, pad_x2, pad_y2)
                ))

        return sub_images

    def adjust_boxes(self, boxes: np.ndarray, confidences: np.ndarray,
                     coords: Tuple[int, int, int, int, int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Adjust box coordinates with more relaxed filtering."""
        pad_x1, pad_y1, orig_x1, orig_y1, orig_x2, orig_y2, _, _ = coords

        # Adjust coordinates back to original image space
        adjusted_boxes = boxes.copy()
        adjusted_boxes[:, [0, 2]] += pad_x1
        adjusted_boxes[:, [1, 3]] += pad_y1

        # Calculate box centers and areas
        centers_x = (adjusted_boxes[:, 0] + adjusted_boxes[:, 2]) / 2
        centers_y = (adjusted_boxes[:, 1] + adjusted_boxes[:, 3]) / 2

        # Calculate overlap with original region
        box_w = adjusted_boxes[:, 2] - adjusted_boxes[:, 0]
        box_h = adjusted_boxes[:, 3] - adjusted_boxes[:, 1]

        # More relaxed filtering: box center or significant portion should be in region
        overlap_threshold = 0.3  # 30% overlap is sufficient
        x_overlap = np.minimum(adjusted_boxes[:, 2], orig_x2) - np.maximum(adjusted_boxes[:, 0], orig_x1)
        y_overlap = np.minimum(adjusted_boxes[:, 3], orig_y2) - np.maximum(adjusted_boxes[:, 1], orig_y1)
        overlap_area = np.maximum(0, x_overlap) * np.maximum(0, y_overlap)
        box_area = box_w * box_h

        valid_mask = ((centers_x >= orig_x1) & (centers_x < orig_x2) &
                      (centers_y >= orig_y1) & (centers_y < orig_y2)) | \
                     (overlap_area / box_area > overlap_threshold)

        # Additional confidence threshold
        valid_mask = valid_mask & (confidences >= self.confidence_threshold)

        return adjusted_boxes[valid_mask], confidences[valid_mask]

    def weighted_box_clustering(self, boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Modified clustering to better preserve small particle detections."""
        if len(boxes) == 0:
            return np.array([])

        clusters = []
        box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        while len(boxes) > 0:
            # Consider both confidence and area for anchor selection
            anchor_idx = scores.argmax()
            anchor_box = boxes[anchor_idx]
            anchor_area = box_areas[anchor_idx]

            # Calculate IoU with all other boxes
            ious = self.calculate_ious(anchor_box, boxes)

            # Adjust IoU threshold based on box size
            size_based_iou = self.iou_threshold * (1.0 if anchor_area > np.median(box_areas) else 0.8)
            mask = ious > size_based_iou

            if mask.sum() > 0:
                # Get all boxes in current cluster
                cluster_boxes = boxes[mask]
                cluster_scores = scores[mask]

                # Modified weighting considering both score and area
                cluster_areas = box_areas[mask]
                area_weights = cluster_areas / cluster_areas.max()
                combined_weights = (cluster_scores * 0.7 + area_weights * 0.3)
                weights = combined_weights / combined_weights.sum()

                merged_box = np.average(cluster_boxes, weights=weights, axis=0)
                clusters.append(merged_box)

            # Remove processed boxes
            boxes = boxes[~mask]
            scores = scores[~mask]
            box_areas = box_areas[~mask]

        return np.array(clusters)

    @staticmethod
    def calculate_ious(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Calculate IoU between one box and an array of boxes."""
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = box_area + boxes_area - intersection

        return intersection / (union + 1e-6)

    def predict_and_merge(self, image_path: str) -> np.ndarray:
        """Processes the image using multiple paddings and returns the merged detection results."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image load failed")

        paddings = [2, 50, 50, 50, 50]  # Adjust padding size based on image size

        all_boxes = []
        all_scores = []

        for pad in paddings:
            self.padding = pad
            sub_images = self.split_image(image)
            for sub_image, coords in sub_images:
                results = self.model.predict(sub_image, verbose=False)[0]
                if len(results.boxes) > 0:
                    boxes = results.boxes.xyxy.cpu().numpy()
                    scores = results.boxes.conf.cpu().numpy()

                    adjusted_boxes, adjusted_scores = self.adjust_boxes(boxes, scores, coords)
                    all_boxes.extend(adjusted_boxes)
                    all_scores.extend(adjusted_scores)

        if not all_boxes:
            return np.array([])

        all_boxes = np.array(all_boxes)
        all_scores = np.array(all_scores)

        # Perform weighted box clustering
        merged_boxes = self.weighted_box_clustering(all_boxes, all_scores)
        return merged_boxes

    def visualize_results(self, image_path: str, save_path: str = None) -> None:
        """
        Visualize particles as circles or rectangles based on aspect ratio.
        Calculate and display area distribution.
        - Circle: width/height ratio > aspect_ratio_threshold
        - Rectangle: width/height ratio <= aspect_ratio_threshold
        Area calculation: 25px = 1μm
        """
        import pandas as pd

        image = cv2.imread(image_path)
        merged_boxes = self.predict_and_merge(image_path)

        if len(merged_boxes) == 0:
            return

        overlay = image.copy()

        # Calculate areas and store shape information
        areas = []
        for box in merged_boxes:
            x1, y1, x2, y2 = map(int, box)
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = min(width, height) / max(width, height)

            if aspect_ratio > self.aspect_ratio_threshold:  # 임계값 사용
                diameter = (width + height) / 2
                area = np.pi * (diameter / 2) ** 2
            else:
                area = width * height

            area_um = (area / (25 * 25))
            areas.append(area_um)

        # Calculate area thresholds for 5 categories
        area_thresholds = np.percentile(areas, [20, 40, 60, 80])

        # Color mapping (BGR format)
        colors = {
            'very_small': (0, 0, 255),  # Red
            'small': (0, 255, 255),  # Yellow
            'medium': (0, 255, 0),  # Green
            'large': (255, 0, 0),  # Blue
            'very_large': (255, 0, 255)  # Purple
        }

        # For area distribution analysis
        area_ranges = {
            'very_small': (0, area_thresholds[0]),
            'small': (area_thresholds[0], area_thresholds[1]),
            'medium': (area_thresholds[1], area_thresholds[2]),
            'large': (area_thresholds[2], area_thresholds[3]),
            'very_large': (area_thresholds[3], float('inf'))
        }

        area_counts = {name: 0 for name in area_ranges.keys()}

        # Draw shapes with size-based colors
        for box, area in zip(merged_boxes, areas):
            x1, y1, x2, y2 = map(int, box)
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = min(width, height) / max(width, height)
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

            # Determine color based on area
            if area < area_thresholds[0]:
                color = colors['very_small']
                area_counts['very_small'] += 1
            elif area < area_thresholds[1]:
                color = colors['small']
                area_counts['small'] += 1
            elif area < area_thresholds[2]:
                color = colors['medium']
                area_counts['medium'] += 1
            elif area < area_thresholds[3]:
                color = colors['large']
                area_counts['large'] += 1
            else:
                color = colors['very_large']
                area_counts['very_large'] += 1

            if aspect_ratio > self.aspect_ratio_threshold:
                # Draw circle
                radius = int((width + height) / 4)
                # cv2.circle(overlay, center, radius, color, -1)
                cv2.circle(overlay, center, radius, color, 1)
            else:
                # Draw rectangle
                # cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)

        # Apply transparency
        alpha = 0.9  # 30% opacity
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        # Create and display area distribution DataFrame
        area_data = []
        total_no = 0
        for name, (min_area, max_area) in area_ranges.items():
            if name == 'very_large':
                range_str = f'>{min_area:.1f}μm²'
            else:
                range_str = f'{min_area:.1f}-{max_area:.1f}μm²'
            area_data.append({
                'Area Range': range_str,
                'Count': area_counts[name]
            })
            total_no = total_no + int(area_counts[name])

        df = pd.DataFrame(area_data)
        print("\nArea Distribution:")
        print(df.to_string(index=False))
        print(f'Total # : {total_no}')

        if save_path:
            cv2.imwrite(save_path, image)

        cv2.imshow("Result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return df, image


# Usage example
detector = ParticleDetector(
    model_path='runs/detect/train22/weights/best.pt',
    grid_size=4,
    iou_threshold=0.08,  # 낮은 IoU 임계값
    padding=5,
    confidence_threshold=0.5,  # 적절한 신뢰도 임계값
    aspect_ratio_threshold=1
)
detector.visualize_results(r"raw_img/10-14-6 Top Pt_q001.jpg")