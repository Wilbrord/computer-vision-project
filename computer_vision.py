#!/usr/bin/env python3
"""Simple YOLOv5 object detection demo for a class exercise.

This script loads a YOLOv5 model, runs object detection on an input image, and
saves an annotated result image.

Example:
    python computer_vision.py
    python computer_vision.py --image sample.jpg --weights yolov5s --conf 0.25
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def load_model(weights: str = "yolov5s"):
    """Load a YOLOv5 model.

    Use either a built-in model name such as yolov5s or a local .pt file path.
    """
    if weights.endswith(".pt"):
        model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=weights,
            source="local",
            force_reload=False,
        )
    else:
        model = torch.hub.load(
            "ultralytics/yolov5",
            weights,
            pretrained=True,
            source="github",
            force_reload=False,
        )

    model.eval()
    return model


def detect_objects(model, image_path: str | Path, confidence_threshold: float = 0.25):
    """Run object detection on an image and return the detections plus saved output path."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    results = model(str(image_path), size=640)
    detections = results.pandas().xyxy[0].copy()
    detections = detections[detections["confidence"] >= confidence_threshold].copy()
    detections["confidence"] = detections["confidence"].round(2)

    output_dir = image_path.parent / "yolov5_results"
    output_dir.mkdir(exist_ok=True)

    save_result = results.save(save_dir=str(output_dir))
    if isinstance(save_result, list) and save_result:
        output_path = save_result[0]
    else:
        output_path = str(output_dir / (image_path.stem + "_yolov5.jpg"))

    return detections, output_path


def print_summary(detections):
    if detections.empty:
        print("No objects detected above the confidence threshold.")
        return

    counts = detections["name"].value_counts().sort_values(ascending=False)
    print("Detected objects:")
    for label, count in counts.items():
        print(f"  - {label}: {count}")

    print("\nDetailed predictions:")
    print(detections[["name", "confidence", "xmin", "ymin", "xmax", "ymax"]].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv5 object detection on an image.")
    default_image = Path(__file__).resolve().parent / "Cat.jpg"
    parser.add_argument(
        "--image",
        type=str,
        default=str(default_image),
        help="Path to the input image file (default: Cat.jpg beside this script).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov5s",
        help="YOLOv5 model name (e.g. yolov5s) or a local .pt path.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Minimum confidence threshold between 0 and 1.")
    parser.add_argument("--output", type=str, default=None, help="Optional path to copy the annotated image to.")
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_model(args.weights)
    detections, output_path = detect_objects(model, args.image, args.conf)

    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if Path(output_path).exists():
            destination.write_bytes(Path(output_path).read_bytes())
            output_path = str(destination)

    print_summary(detections)
    print(f"\nAnnotated image saved to: {output_path}")


if __name__ == "__main__":
    main()
