import ultralytics as ut
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_yolo():
    yolo = ut.YOLO("yolo11n.pt")
    yolo.train(
        data="./data/midv_all/data/yolo_idcard.yaml",
        epochs=200,
        save=True,
        project="./yolo11_finetune_1",
        exist_ok=True,
        pretrained=True,
        classes=[0, 1],
        device=0,
        save_period=1,
        hsv_h=0.25,
        hsv_s=0.85,
        hsv_v=0.7,
        degrees=0.2,
        translate=0.2,
        shear=0.1,
        flipud=0.5,
        copy_paste=0.1,
    )


if __name__ == "__main__":
    train_yolo()
