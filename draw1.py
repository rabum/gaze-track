#检查人体姿态识别和头部姿态识别是否是均为两个结果

import sys
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())

import os
import cv2
import numpy as np
from mmpose.apis import MMPoseInferencer
from gaze import YOLOv8_face
from gaze import L2CSNet
from utils.mmpose import inference_video, calculate_face_bbox, get_head_pose, plot_3axis_Zaxis
from utils.torch_utils import select_device
from utils.general import check_img_size, scale_coords, non_max_suppression
from utils.datasets import LoadImages
from models.experimental import attempt_load

def draw(video_path):

    results = inference_video(video_path, vis=True)
    mp4_name = video_path.split('/')[-1]
    labeled_video_path = f'vis_results/{mp4_name}'
    labeled_video = cv2.VideoCapture(labeled_video_path)
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("can not open video")
        exit()
    for i, result in enumerate(results):
        human_count = len(result['predictions'][0])
        ret, frame = video.read()
        _, labeled_frame = labeled_video.read()
        frame, head_pose = get_head_pose(frame)
        for pose, box in head_pose:
            pitch, yaw, roll = pose
            x1, y1, x2, y2 = box
            labeled_frame = plot_3axis_Zaxis(labeled_frame, yaw, pitch, roll, tdx=(x1+x2)/2, tdy=(y1+y2)/2, 
                size=max(y2-y1, x2-x1)*0.8, thickness=2)
    
        cv2.imwrite(f"temp/{i}.jpg", labeled_frame)
    video.release()

if __name__ == '__main__':

    video_path = 'open_door/right-3-26/cb3302c099168781e6cc8e26d87efabe.mp4'
    #video_path = 'wrong-3-26/1ab10af0504baaf810ae90f58da3a79b.mp4'
    draw(video_path)