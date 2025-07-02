'''
This file has a class to load the trained model and use
it to predict the lables on an image/frame or a video.
'''

from ultralytics import YOLO
from PIL import Image
import torch

class PPEDetector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = YOLO(self.model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.to(self.device)
    
    def predict_frame(self, image):
        '''
        Description:

        Takes PIL or cv2 or path to an image and return PIL image with predictied annotations in it.
        '''

        result = self.model(image)
        return Image.fromarray(result[0].plot()[:, :, ::-1])

    def predict_video(self, video_path):
        '''
        Description:

        Takes video path and save a new video with predicted annotations in it.
        
        The video is saved in "./runs/video_output".

        Note: display each prediction frame by frame use predict_image and pass each frame to it.
        '''

        print(self.model(video_path, save=True, name="./video_output"))
