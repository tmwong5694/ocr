import argparse
import cv2
import yaml
from pathlib import Path
from ultralytics import YOLO

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def detect_cats_and_dogs(
        image_path: str | Path,
        model_path: str | Path,
        output_path: str | Path,
        confidence_threshold: float = 0.50,
        target_classes: list = None
):
    if target_classes is None:
        target_classes = [15, 16]

    # Store the model in the models directory to keep the workspace clean
    if isinstance(model_path, str):
        model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    model = YOLO(model_path if model_path.exists() else "yolov8n.pt")

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    results = model(img)

    COCO_CLASSES = {15: 'Cat', 16: 'Dog'}

    for result in results:
        boxes = result.boxes

        for box in boxes:
            # Get the class ID and confidence score
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            # Only process if the detected object is a target class and meets threshold
            if cls_id in target_classes and conf >= confidence_threshold:
                # Get bounding box coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                class_name = COCO_CLASSES.get(cls_id, f"Class_{cls_id}")
                label = f"{class_name} {conf:.2f}"

                # Pick a color (Blue for Cat, Green for Dog) - BGR format in OpenCV
                color = (255, 0, 0) if cls_id == 15 else (0, 255, 0)

                # Draw the rectangle around the object, corner1 (x1, y1) and corner2 (x2, y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2)

                text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
                cv2.putText(
                    img,
                    label,
                    (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6,
                    color=color,
                    thickness=2
                )
    cv2.imwrite(str(output_path), img)
    print(f"Saved detection result to {output_path}")


def cli_main():
    parser = argparse.ArgumentParser(description="Run YOLO object detection on an image")
    parser.add_argument("--config", default="ml/.config/detect.yaml", help="Path to the config file (detect.yaml)")
    args = parser.parse_args()

    config = load_config(args.config)

    # Extract params from config
    thresh = config.get('model', {}).get('confidence_threshold', 0.5)
    classes = config.get('inference', {}).get('target_classes', [15, 16])
    model_weight = Path(config.get('model', {}).get('weights', 'ml/src/ml_pipeline/models/yolov8n.pt'))
    image_path = config.get('image')
    
    if not image_path:
        raise ValueError("Image path not specified in detect.yaml config file")
    
    # Generate a sensible output path based on the input name
    out_path = Path(image_path).stem + "_detected.jpg"

    detect_cats_and_dogs(
        image_path=image_path,
        model_path=model_weight,
        output_path=out_path,
        confidence_threshold=thresh,
        target_classes=classes
    )

if __name__ == "__main__":
    cli_main()
