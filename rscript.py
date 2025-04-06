import numpy as np
import torch
import ultralytics as ut

yolo = ut.YOLO("./yolo11_finetune/train/weights/best.pt")
mg = "/root/code/data/PAD-IDCARD-2025/bonafide-ijcb2025PAD-IDcard/bonafide/0.png"
results = yolo([mg])

for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk
