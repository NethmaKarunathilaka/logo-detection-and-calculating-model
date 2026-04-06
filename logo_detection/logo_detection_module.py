import os
import cv2
from ultralytics import YOLO


def yolov8_logo_detection(
    model_path,
    image_path,
    save_result=False,
    conf=0.12,
    iou=0.6,
    max_det=1000,
    imgsz=1280,
    augment=True,
):
    """
    Detects logos in an image using the YOLOv8 model.

    This function loads a YOLOv8 model from a specified path, processes an image for logo detection,
    and optionally saves the result with bounding boxes drawn around detected logos.

    Parameters:
    model_path (str): The file path to the pre-trained YOLOv8 model.
    image_path (str): The file path of the image in which logos are to be detected.
    save_result (bool, optional): If True, the image with detected logos (bounding boxes drawn)
                                  is saved. Default is False.
    conf (float, optional): Confidence threshold for detection (0.0 to 1.0). Lower values detect more logos.
                           Default is 0.12 for higher recall on small/blurred logos.
    iou (float, optional): NMS IoU threshold (0.0 to 1.0). Higher values keep more overlapping candidates.
                           Default is 0.6 to retain nearby small-logo predictions.
    max_det (int, optional): Maximum number of detections per image. Default is 1000.
    imgsz (int, optional): Inference image size. Larger values help detect small logos.
                           Default is 1280.
    augment (bool, optional): Enables test-time augmentation for better robustness on blur/scale.
                              Default is True.

    Returns:
    list: A list of bounding boxes for each detected logo. Each bounding box is represented as
          a list of coordinates [x1, y1, x2, y2], where (x1, y1) is the top-left corner, and
          (x2, y2) is the bottom-right corner of the bounding box.
    """
    # Load the YOLO model
    model = YOLO(model_path)

    # Use larger image size and optional TTA to improve recall for tiny/blurred logos.
    results = model(
        image_path,
        conf=conf,
        iou=iou,
        max_det=max_det,
        imgsz=imgsz,
        augment=augment,
    )
    results = (results[0].boxes).xyxy
    bboxes = []
    for box in results:
        bboxes.append(box.cpu().tolist())

    # Save the image with drawn bounding boxes, if required
    if save_result:
        image = cv2.imread(image_path)
        filename = image_path.split("/")[-1].split(".")[0]
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            # Draw a rectangle around the bounding box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Save image with bounding boxes
        os.makedirs("./results", exist_ok=True)
        cv2.imwrite(f'results/{filename}_detected_logo.png', image)

    return bboxes

