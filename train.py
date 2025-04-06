import ultralytics as ut
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_yolo():
    yolo = ut.YOLO("yolo11n.pt")
    yolo.train(
        data="./data/midv_all/data/yolo_idcard.yaml",
        epochs=100,
        save=True,
        project="./yolo11_finetune",
        exist_ok=True,
        pretrained=True,
        classes=[0,1],
        device=0,
        save_period=1,
    )


if __name__ == "__main__":
    train_yolo()
