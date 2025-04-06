import json
import shutil
import os
from os.path import join as pjoin
from PIL import Image
from tqdm import tqdm


def val():
    trainds = "./data/midv_all/data/midv2019_original"
    yolods = "./data/midv_all/data/val"
    with open(trainds.replace("_original", "") + "_coco.json", "r") as fp:
        data = json.load(fp)

    for image, annotation in tqdm(zip(data["images"], data["annotations"])):
        fname = image["file_name"]
        image_height = image["height"]
        image_width = image["width"]
        bbox = annotation["bbox"]
        x_min, y_min, width, height = bbox
        x_min, y_min, width, height = (
            float(x_min),
            float(y_min),
            float(width),
            float(height),
        )
        category_id = annotation["category_id"]
        center_x_scaled = x_min + (width * 0.5)
        center_y_scaled = y_min + (height * 0.5)
        width_scaled = width / image_width
        height_scaled = height / image_height
        noramlised_bbox = (
            category_id,
            center_x_scaled / image_width,
            center_y_scaled / image_height,
            width_scaled,
            height_scaled,
        )

        oname = fname.replace("/", "_").replace(".tif", ".png")
        os.makedirs(os.path.dirname(pjoin(yolods, "images", oname)), exist_ok=True)
        os.makedirs(
            os.path.dirname(pjoin(yolods, "labels", oname.replace(".png", ".txt"))),
            exist_ok=True,
        )
        Image.open(pjoin(trainds, fname)).save(pjoin(yolods, "images", oname))

        with open(pjoin(yolods, "labels", oname.replace(".png", ".txt")), "w+") as fp:
            fp.write(" ".join([str(x) for x in noramlised_bbox]))


def train():
    trainds = "./data/midv_all/data/midv500_original"
    yolods = "./data/midv_all/data/train"
    with open(trainds.replace("_original", "") + "_coco.json", "r") as fp:
        data = json.load(fp)

    for image, annotation in tqdm(zip(data["images"], data["annotations"])):
        fname = image["file_name"]
        oname = fname.replace("/", "_").replace(".tif", ".png")
        image_height = image["height"]
        image_width = image["width"]
        bbox = annotation["bbox"]
        x_min, y_min, width, height = bbox
        x_min, y_min, width, height = (
            float(x_min),
            float(y_min),
            float(width),
            float(height),
        )
        category_id = annotation["category_id"]
        center_x_scaled = x_min + (width * 0.5)
        center_y_scaled = y_min + (height * 0.5)
        width_scaled = width / image_width
        height_scaled = height / image_height
        noramlised_bbox = (
            category_id,
            center_x_scaled / image_width,
            center_y_scaled / image_height,
            width_scaled,
            height_scaled,
        )
        # os.makedirs(os.path.dirname(pjoin(yolods, "images", oname)), exist_ok=True)
        # os.makedirs(
        #     os.path.dirname(pjoin(yolods, "labels", oname.replace(".png", ".txt"))),
        #     exist_ok=True,
        # )
        # Image.open(pjoin(trainds, fname)).save(pjoin(yolods, "images", oname))

        with open(pjoin(yolods, "labels", oname.replace(".png", ".txt")), "w+") as fp:
            fp.write(" ".join([str(x) for x in noramlised_bbox]))


if __name__ == "__main__":
    val()
    train()
