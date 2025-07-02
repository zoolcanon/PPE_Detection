FROM python:3.12.4

COPY . .

WORKDIR /app

RUN apt-get update && apt-get upgrade -y && apt-get install -y libgl1-mesa-glx
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install pandas timm opencv-python scikit-learn ultralytics torchmetrics[detection]