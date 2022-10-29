import torch
import numpy as np
import cv2
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, fasterrcnn_resnet50_fpn
from VisualizeDetection import visualizaion_but_show, visualizaion
from utils import image_array2tensor, tensor2cv2image
from PIL import Image


@torch.no_grad()
def demo():
    model = ssdlite320_mobilenet_v3_large(pretrained=True).eval()
    # Capture OBS Virtual Camera
    cap = cv2.VideoCapture(0)
    # Detectiuon Loop
    while cap.isOpened():
        ret, frame = cap.read()
        x = image_array2tensor(frame)
        # Make detections
        results = model(x)
        visualizaion_but_show(results, frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


@torch.no_grad()
def visualize_one_image(image_path='./1.jpg'):
    from torchvision import transforms
    to_tensor = transforms.ToTensor()
    img = Image.open(image_path)
    x = to_tensor(img)
    model = ssdlite320_mobilenet_v3_large(pretrained=True).eval()
    source = cv2.imread(image_path)
    visualizaion(model([x]), source)


if __name__ == '__main__':
    demo()
    # visualize_one_image()
