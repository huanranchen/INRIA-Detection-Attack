import torch
import numpy as np
import cv2
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, fasterrcnn_resnet50_fpn
from VisualizeDetection import visualizaion_but_show
from utils import image_array2tensor, tensor2cv2image

@torch.no_grad()
def demo():
    model = fasterrcnn_resnet50_fpn(pretrained=True).eval()

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

demo()
