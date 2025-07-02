from ultralytics import YOLO
from ultralytics.utils.ops import xyxy2xywhn
from PIL import Image
import torch
import os

classes = ["person", "vest", "glass", "head", "red", "yellow", "blue", "white"]
device = "cuda"  if torch.cuda.is_available() else "cpu"

def info(labels_path):
    values = {}
    files = os.listdir(labels_path)
    for file in files:
        if os.path.splitext(file)[1] == ".txt":
            f = open(f"{labels_path}/{file}", "r").read()
            for i in f.split('\n'):
                if i != '':
                    value = i.split()[0]
                    if value not in values:
                        values.update({value: 1})
                    else:
                        values[str(value)] += 1
    output = sorted(values.items(), key= lambda x: x[1])
    output = [(classes[int(x[0])], x[1]) for x in output]
    print(output)
                

def train():
    model = YOLO("yolo12l.pt")
    model.to(device)
    result = model.train(
        data="config.yml",
        epochs=100,
        batch=16,
        lr0=0.00001,
        lrf=0.0000001,
        optimizer="NAdam",
        box=7.5,
        patience=7,
        cos_lr=True,
        momentum=0.937,
        dropout=0.4,
        weight_decay=0.01,
        erasing=0,
        degrees=5,
        mosaic=0,
        hsv_s=0.7,
        workers=10)
    #0.00001
    # results = model.val()

def val(model_path):
    model = YOLO(model_path)
    model.to(device)
    model.val()

def fine_tune(model_path):
    model = YOLO(model_path)
    model.to(device)
    result = model.tune(
        data="config.yml",
        iterations=100,
        epochs=6,
        batch=16,
        lr0=0.00001,
        lrf=0.0000001,
        optimizer="NAdam",
        box=7.5,
        patience=5,
        cos_lr=True,
        momentum=0.937,
        dropout=0.4,
        weight_decay=0.01,
        workers=10)

def use_model(model_path, images_path):
    def convert_to_yolo_format(cls, box, img_width, img_height):
        xywh = xyxy2xywhn(box, img_width, img_height)
        return cls, xywh[0][0], xywh[0][1], xywh[0][2], xywh[0][3]
    
    images = [os.path.join(images_path, x) for x in os.listdir(images_path)]
    model = YOLO(model_path)
    model.to(device)
    for image in images:
        results = model(image)
        # Image.fromarray(results[0].plot()).save("./out.jpg")

        for result in results:
            text = ""
            
            for box in result.boxes:
                converted = convert_to_yolo_format(int(box.cls.item()), box.xyxy, 640, 640)
                if converted[1] <= 1 and converted[2] <= 1 and converted[3] <= 1 and converted[4] <= 1:
                    if text != "":
                        text = text + f"{converted[0]} {converted[1]:6f} {converted[2]:6f} {converted[3]:6f} {converted[4]:6f}\n"
                    else:
                        text = f"{converted[0]} {converted[1]:6f} {converted[2]:6f} {converted[3]:6f} {converted[4]:6f}\n"

            if text != "":
                print(text, '\n')
                output_file = f"{os.path.splitext(result.path.split('/')[-1])[0]}.txt"
                file = open(f"./out_labels/{output_file}", 'w')
                file.write(text)
                file.close()

if __name__ == "__main__":
    # use_model("./runs/detect/best-yolo11x/weights/best.pt", "./unstructured_images/PPE.v1i.yolov11/train/images/")

    train()
    # fine_tune("yolo11x.pt")
