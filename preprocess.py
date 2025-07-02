import os
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import PIL
import xml.etree.ElementTree as ET
import shutil
from sklearn.model_selection import train_test_split
import numpy as np
from multiprocessing import Pool, cpu_count
import cupy as cp
import time
import hashlib
import pandas as pd
import random

class XMLToYOLO:
    def __init__(self):
        self.dataset_path = "./datasets/CHVG-Dataset"
        self.train_dir = f"{self.dataset_path}/images/train"
        self.test_dir = f"{self.dataset_path}/images/test"
        self.xml_labels_dir = f"{self.dataset_path}/xml_labels"
        self.xml_train_labels_dir = f"{self.xml_labels_dir}/train"
        self.xml_test_labels_dir = f"{self.xml_labels_dir}/test"
        self.text_labels_dir = f"{self.dataset_path}/labels"
        self.text_train_labels_dir = f"{self.text_labels_dir}/train"
        self.text_test_labels_dir = f"{self.text_labels_dir}/test"

        self.files = [name for name in os.listdir(f"{self.dataset_path}") if os.path.isfile(os.path.join(f"{self.dataset_path}", name))]
        self.dataset_len = len(self.files)

        self.train = self.files[:self.dataset_len - int(self.dataset_len * 0.2)]
        self.test = self.files[self.dataset_len - int(self.dataset_len * 0.2):]

    def move_files(self):
        for i in self.train:
            if ".jpg" in i:
                os.rename(f"{self.dataset_path}/{i}", f"{self.train_dir}/{i}")
            else:
                os.rename(f"{self.dataset_path}/{i}", f"{self.xml_train_labels_dir}/{i}")
        for i in self.test:
            if ".jpg" in i:
                os.rename(f"{self.dataset_path}/{i}", f"{self.test_dir}/{i}")
            else:
                os.rename(f"{self.dataset_path}/{i}", f"{self.xml_test_labels_dir}/{i}")

    def convert_to_yolo_format(self, xml_file_path, classes):
        # Parse the XML file
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        # Extract image size
        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)
        
        yolo_data = []

        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in classes:
                continue
            
            class_id = classes.index(class_name)
            
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            # Convert to YOLO format
            x_center = (xmin + xmax) / 2.0 / width
            y_center = (ymin + ymax) / 2.0 / height
            bbox_width = (xmax - xmin) / width
            bbox_height = (ymax - ymin) / height
            
            yolo_data.append(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}")
        
        return yolo_data

    def make_yolo_dir_format(self):
        os.makedirs(f"{self.dataset_path}/images")
        os.makedirs(self.train_dir)
        os.makedirs(self.test_dir)
        os.makedirs(self.xml_labels_dir)
        os.makedirs(self.xml_train_labels_dir)
        os.makedirs(self.xml_test_labels_dir)

        self.move_files()

        classes = ["person", "vest", "glass", "head", "red", "yellow", "blue", "white"]

        os.makedirs(self.text_labels_dir)
        os.makedirs(self.text_train_labels_dir)
        os.makedirs(self.text_test_labels_dir)
        def save_text_files(xml_dir_path, text_label_dir_path):
            for file_name in os.listdir(xml_dir_path):
                yolo_format = self.convert_to_yolo_format(f"{xml_dir_path}/{file_name}", classes)
                text = ""
                for line in yolo_format:
                    if yolo_format.index(line) == 0:
                        text = line + '\n'
                    else:
                        text = text + line + '\n'

                file = open(f"{text_label_dir_path}/{file_name.split(".xml")[0]}.txt", 'w')
                file.write(text)
                file.close()

        save_text_files(xml_dir_path=self.xml_train_labels_dir, text_label_dir_path=self.text_train_labels_dir)
        save_text_files(xml_dir_path=self.xml_test_labels_dir, text_label_dir_path=self.text_test_labels_dir)




class DataInfo:
    def __init__(self, labels_path):
        self.labels_path = labels_path
        self.files = os.listdir(self.labels_path)
        self.output = None
        self.classes = ["person", "vest", "glass", "head", "red", "yellow", "blue", "white"]

        values = {}
        for file in self.files:
            if os.path.splitext(file)[1] == ".txt":
                try:
                    f = open(f"{self.labels_path}/{file}", "r").read()
                except FileNotFoundError:
                    continue
                
                for i in f.split('\n'):
                    if i != '':
                        value = i.split()[0]
                        if value not in values:
                            values.update({value: 1})
                        else:
                            values[str(value)] += 1
        self.output = sorted(values.items(), key= lambda x: x[1])
        self.output = [(self.classes[int(x[0])], x[1]) for x in self.output]
        indexes = [x[0] for x in self.output]
        instances = [x[1] for x in self.output]
        self.output = pd.DataFrame(instances, columns=["number of instances"], index=indexes)

    def print_info(self):
        print(self.output)
        print("Total labels", len(self.files))
    
    def get_info(self):
        return self.output.to_dict()["number of instances"]

    def plot(self, exclude:list=None):
        if exclude:
            self.output = self.output.drop(exclude)
        self.output.plot.bar()

class Preprocess:
    def __init__(self, images_path, labels_path, dest_path):
        self.images_path = images_path
        self.labels_path = labels_path
        self.dest_path = dest_path
        self.images = [x for x in os.listdir(self.images_path) if os.path.isfile(os.path.join(self.images_path, x))]
        self.labels = [x for x in os.listdir(self.labels_path) if os.path.isfile(os.path.join(self.labels_path, x))]
        self.full_image_files_path = [os.path.join(self.images_path, x) for x in self.images]
        self.full_label_file_paths = [os.path.join(self.labels_path, x) for x in self.labels]
        self.classes = ["person", "vest", "glass", "head", "red", "yellow", "blue", "white"]

        if "desktop.ini" in self.images:
            self.images.remove("desktop.ini")
        if "desktop.ini" in self.labels:
            self.labels.remove("desktop.ini")
        

        try:
            os.makedirs(f"{self.dest_path}/images")
            os.makedirs(f"{self.dest_path}/labels")
        except FileExistsError:
            pass

    @staticmethod
    def resize(images_path, dest_path:str, shape:tuple):
        images = [f"{images_path}/{x}" for x in os.listdir(images_path) if os.path.splitext(x)[1] == ".png" or os.path.splitext(x)[1] == '.jpg']

        for image in images:
            img = Image.open(image)
            if img.size != shape:
                img = img.resize(shape) 
            img.save(f"{dest_path}/{image.split('/')[-1]}")

    def copy_images_with_labels(self):
        for img in self.images:
            image_name = os.path.splitext(img)[0]

            shutil.copy(f"{self.images_path}/{img}", f"{self.dest_path}/images/")
            shutil.copy(f"{self.labels_path}/{image_name}.txt", f"{self.dest_path}/labels/")
    
    def move_images_with_labels(self):
        for img in self.images:
            image_name = os.path.splitext(img)[0]

            os.rename(f"{self.images_path}/{img}", f"{self.dest_path}/images/")
            os.rename(f"{self.labels_path}/{image_name}.txt", f"{self.dest_path}/labels/")
    
    def copy_images_with_labels_and_execlude(self, execlude_images_path):
        '''
        Copy images and labels and execlude some images.

        args:
            exclude_images_path: the images you don't want to have.
        '''
        execlude_imgs = os.listdir(execlude_images_path)

        for image in self.images:
            if image not in execlude_imgs:
                # print(image)
                label_file_name = f"{os.path.splitext(image)[0]}.txt"
                try:
                    shutil.copy(f"{self.images_path}/{image}", f"{self.dest_path}/images/{image}")
                    shutil.copy(f"{self.labels_path}/{label_file_name}", f"{self.dest_path}/labels/{label_file_name}")
                except FileNotFoundError as e:
                    print(e)
        print("process is done...")
    
    def copy_images_with_labels_until(self, until_this_file):

        for image in self.images:
            image_name, _ = os.path.splitext(image)
            label = f"{image_name}.txt"
            label_path = os.path.join(self.labels_path,label)
            image_path = os.path.join(self.images_path, image)
            new_dest_images = os.path.join(self.dest_path, "images")
            new_dest_labels = os.path.join(self.dest_path, "labels")
            
            shutil.copy(label_path, new_dest_labels)
            shutil.copy(image_path, new_dest_images)

            if image_name == os.path.splitext(until_this_file)[0]:
                break

    def split_data(self, ratio, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(self.images, self.labels, test_size=ratio, random_state=random_state)

        try:
            os.makedirs(os.path.join(self.dest_path, "images", "train"))
            os.makedirs(os.path.join(self.dest_path, "images", "test"))
            os.makedirs(os.path.join(self.dest_path, "labels", "train"))
            os.makedirs(os.path.join(self.dest_path, "labels", "test"))
        except FileExistsError:
            pass

        for image in X_train:
            image_name, _ = os.path.splitext(image)
            label_name = f"{image_name}.txt"
            image_path = os.path.join(self.images_path, image)
            label_path = os.path.join(self.labels_path, label_name)

            shutil.copy(image_path, os.path.join(self.dest_path, "images/train", image))
            shutil.copy(label_path, os.path.join(self.dest_path, "labels/train", label_name))
            

        for image in X_test:
            image_name, _ = os.path.splitext(image)
            label_name = f"{image_name}.txt"
            image_path = os.path.join(self.images_path, image)
            label_path = os.path.join(self.labels_path, label_name)
            
            try:
                shutil.copy(image_path, os.path.join(self.dest_path, "images/test", image))
                shutil.copy(label_path, os.path.join(self.dest_path, "labels/test", label_name))
            except FileNotFoundError:
                print(image_path)
                print(os.path.exists(image_path))

    def horizontal_flip(self):
        for image in self.images:
            new_label = ""
            
            image_path = os.path.join(self.images_path, image)
            image_file_name = image
            try:
                image = Image.open(image_path)
            except PIL.UnidentifiedImageError:
                continue
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            image.save(f"{self.dest_path}/images/flipped_{os.path.splitext(image_file_name)[0]}{os.path.splitext(image_file_name)[1]}")

            label_file_name = f"{os.path.splitext(image_file_name)[0]}.txt"
            label_path = os.path.join(self.labels_path, label_file_name)
            file = open(label_path, "r")
            label = file.read().split('\n')
            
            for i in label:
                text = i.split()
                if len(text) > 0:
                    x_new = 1 - float(text[1])
                    new_label = new_label + f"{text[0]} {x_new:6f} {text[2]} {text[3]} {text[4]}\n"

            open(f"{self.dest_path}/labels/flipped_{label_file_name}", "w").write(new_label)
            file.close()

    def change_brightness(self, factor):
        for image in self.images:
            image_name, image_extention = os.path.splitext(image)
            label_name = f"{image_name}.txt"
            new_name = f"brightened{factor}_{image_name}"
            new_image_name = f"{new_name}{image_extention}"
            new_label_name = f"{new_name}.txt"
            try:
                img = Image.open(os.path.join(self.images_path, image))
            except PIL.UnidentifiedImageError:
                continue
            img = ImageEnhance.Brightness(img)
            enhanced_img = img.enhance(factor)
            
            enhanced_img.save(os.path.join(self.dest_path, "images", new_image_name))
            shutil.copy(os.path.join(self.labels_path, label_name), os.path.join(self.dest_path, "labels", new_label_name))

    def blur(self, radius):
        for image in self.images:
            image_name, image_extention = os.path.splitext(image)
            label_name = f"{image_name}.txt"
            new_name = f"blurred_{image_name}"
            new_image_name = f"{new_name}{image_extention}"
            new_label_name = f"{new_name}.txt"

            try:
                img = Image.open(os.path.join(self.images_path, image))
            except PIL.UnidentifiedImageError:
                continue

            img = img.filter(ImageFilter.GaussianBlur(radius))

            img.save(os.path.join(self.dest_path, "images", new_image_name))
            shutil.copy(os.path.join(self.labels_path, label_name), os.path.join(self.dest_path, "labels", new_label_name))

    def add_gaussian_noise(self, mean=1, std=25):
        for image in self.images: 
            img_name, ext = os.path.splitext(image)
            label_name = f"{img_name}.txt"
            new_name = f"noisy_{img_name}"
            new_img_name = f"{new_name}{ext}"
            new_label_name = f"{new_name}.txt"

            img = Image.open(os.path.join(self.images_path, image))
            img_array = np.array(img)
            noise = np.random.normal(mean, std, img_array.shape)
            noisy_img = img_array + noise
            noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

            img = Image.fromarray(noisy_img)
            img.save(os.path.join(self.dest_path,"images", new_img_name))
            shutil.copy(os.path.join(self.labels_path, label_name), os.path.join(self.dest_path, "labels", new_label_name))

    def random_corp_class(self, image:Image, label_lines:list, classes:list, class_name, current_num_instances, limit):
        width, height = image.size
        annotations = []
        new_labels = []

        draw = ImageDraw.Draw(image)

        for line in label_lines:
            parts = line.strip().split()
            class_id = int(parts[0])

            if classes[class_id] == class_name:
                annotations.append(line)
            else:
                new_labels.append(line)

        try:
            annotations = random.sample(annotations, random.randint(1, len(annotations) - 1))
        except ValueError:
            pass

        new_labels = list(set(label_lines) - set(annotations))

        if annotations:
            for line in annotations:
                if not (current_num_instances <= limit):
                    parts = line.strip().split()
                    class_id = int(parts[0])

                    if classes[class_id] == class_name:
                        x_center, y_center, w, h = map(float, parts[1:])

                        x1 = int((x_center - w / 2) * width)
                        y1 = int((y_center - h / 2) * height)
                        x2 = int((x_center + w / 2) * width)
                        y2 = int((y_center + h / 2) * height)

                        # Blackout the region
                        draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))
                        current_num_instances -= 1
                else:
                    break
        
        return image, new_labels, current_num_instances
    
    def down_sample(self, class_instances:dict, limit:int):
        new_class_instances = dict()
        instances_counter = dict()
        # Now it's going on all images to check only one class, you need to change this later
        # to make it to go across all classes.

        for i in class_instances:
            if class_instances[i] > limit and i != "person":
                new_class_instances[i] = class_instances[i]
                instances_counter[i] = class_instances[i]

        for image in self.images:
            image_name = image
            label_name = f"{os.path.splitext(image)[0]}.txt"
            full_image_path = os.path.join(self.images_path, image)
            full_label_path = os.path.join(self.labels_path, label_name)
            
            with open(f"{full_label_path}", 'r') as f:
                    lines = f.readlines()

            pil_image = Image.open(full_image_path)
            new_labels = lines

            for class_name in new_class_instances:
                if not (instances_counter[class_name] <= limit):
                    pil_image, new_labels, instances_counter[class_name] = self.random_corp_class(pil_image,
                                                                                    new_labels,
                                                                                    self.classes,
                                                                                    class_name,
                                                                                    instances_counter[class_name],
                                                                                    limit)
            # After corping all random selected instances of all classes save the image and labels
            str_new_labels = [i.replace('\n', '').strip() for i in new_labels]
            str_new_labels = '\n'.join(str_new_labels)

            # Save processed image and labels
            pil_image.save(os.path.join(self.dest_path, "images", image_name))
            with open(os.path.join(self.dest_path, "labels", label_name), 'w') as f:
                f.write(str_new_labels)
                f.close()

    def augmentation(self):
        # self.horizontal_flip()
        self.copy_images_with_labels()
        # preprocess = Preprocess(os.path.join(self.dest_path, "images"), os.path.join(self.dest_path, "labels"), f"{self.dest_path}2")
        # preprocess.change_brightness(1.3)
        # Preprocess(os.path.join(self.dest_path, "images"), os.path.join(self.dest_path, "labels"), self.dest_path).blur(2)
        # preprocess = Preprocess(os.path.join(self.dest_path, "images"), os.path.join(self.dest_path, "labels"), f"{self.dest_path}3")
        self.add_gaussian_noise()
    
    def delete_images_with_same_pixels(self, test_images_path, test_labels_path):
        def compute_image_hash(image_path):
            with open(image_path, 'rb') as f:
                image_data = f.read()
            return hashlib.md5(image_data).hexdigest()

        test_images_file_path = [os.path.join(test_images_path, x) for x in os.listdir(test_images_path)]
        test_labels_file_path = [os.path.join(test_labels_path, x) for x in os.listdir(test_labels_path)]

        for full_image_path in self.full_image_files_path:
            image_hash = compute_image_hash(full_image_path)
        
            for i in test_images_file_path:
                image2_hash = compute_image_hash(i)
                if full_image_path != i and image_hash == image2_hash:
                    print(full_image_path)
                    print(i + "\n")

        with Pool(8) as p:
            p.map()


if __name__ == "__main__":
    data_info = DataInfo("./datasets/v5/CHVG-Dataset/labels/train").get_info()
    preprocess = Preprocess("./new/images/", "./new/labels/", "./new/preprocessed")
    preprocess.down_sample(data_info, 6000)
    DataInfo("./new/preprocessed/labels").print_info()
    