"""
Command-line interface for YOLOv8-based Logo Detection.

This script performs logo detection using the YOLOv8 model on an input image.
It takes the path to the YOLOv8 model, the input image, and an optional flag to save the result image.

Usage:
python main_detection_yolov8.py --model path/to/model.pt --image path/to/image --save-result

Arguments:
--model          : Path to the YOLOv8 model file.
--image          : Path to the input image file.
--save-result    : Flag to save the result image with bounding boxes.
"""

import argparse

from logo_detection.logo_detection_module import yolov8_logo_detection


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Logo Detection")
    parser.add_argument("--model", type=str, default='weights/Logo_Detection_Yolov8.pt'
                        , help="Path to YOLOv8 trained model")
    parser.add_argument("--image", type=str, default='test_images/test.jpg', help="Path to input image")
    parser.add_argument("--save-result", action="store_true", help="Save the result image with bounding boxes")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (0.0-1.0). Lower detects more logos")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold (0.0-1.0). Lower keeps more overlapping boxes")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum number of detections per image")

    args = parser.parse_args()

    model_path = args.model
    image_path = args.image
    save_result = args.save_result
    conf = args.conf
    iou = args.iou
    max_det = args.max_det

    # Perform logo detection and get bounding boxes
    bboxes = yolov8_logo_detection(model_path, image_path, save_result, conf=conf, iou=iou, max_det=max_det)
    print("bounding-boxes list :")
    print(bboxes)

    if save_result:
        print(f"Bounding boxes saved to ./results/{image_path.split('/')[-1].split('.')[0]}_detected_logo.png")


if __name__ == "__main__":
    main()