"""
ID_CARD Detector inference script.
This script is used to perform inference on images using a YOLOv11 model.
It loads the model, and performs inference on a given image or a list of images.
"""

import argparse
from typing import Any
from PIL import Image
import numpy as np
import ultralytics as ut

parser = argparse.ArgumentParser(description="YOLOv8 Inference")
parser.add_argument(
    "--image",
    type=str,
    help="Image path",
)
parser.add_argument(
    "--output",
    type=str,
    default="./output.jpg",
    help="Output image path",
)


def inference(input_image: str, output_image: str) -> Any:
    """
    Crops the ID card from the input image and saves it to the output path.
    """
    yolo = ut.YOLO("./yolo11_finetune/train/weights/best.pt")
    image = Image.open(input_image)
    results = yolo.predict([image], batch=True, classes=[1], verbose=False)
    image = np.array(image)

    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        probs = result.probs  # Probs object for classification outputs
        print("BOX", boxes)
        classid = boxes.cls.detach().cpu().tolist()[0]
        if not classid:
            print("No ID card detected")
            return
        x1, y1, x2, y2 = boxes.xyxy.detach().cpu().tolist()[0]
        cropped_image = image[int(y1) : int(y2), int(x1) : int(x2)]
        pil_image = Image.fromarray(cropped_image)
        pil_image.save(output_image)


if __name__ == "__main__":
    args = parser.parse_args()
    input_image = args.image
    output_image = args.output
    inference(input_image, output_image)
