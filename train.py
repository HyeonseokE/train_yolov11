from ultralytics import YOLO
import os
from setproctitle import setproctitle

setproctitle("aa007878")
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


if __name__ == "__main__":
    # Load a COCO-pretrained YOLO11n model
    model = YOLO("yolo11x.pt")

    # Train the model on the COCO8 example dataset for 100 epochs
    results = model.train(data="./coco.yaml", epochs=45, batch=32, imgsz=1280, val=False, device=0,1,2,3,4,5,6,7)
    