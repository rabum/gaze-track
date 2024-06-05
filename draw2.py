# 在人体姿态识别和头部姿态识别均为两个结果的情况先，画出观察者的视线方向以及与开门者之间的方向向量
import math
import sys
import cv2
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())

import os
import cv2
import numpy as np
from mmpose.apis import MMPoseInferencer
from tqdm import tqdm
from gaze import YOLOv8_face
from gaze import L2CSNet
from utils.mmpose import inference_video, calculate_face_bbox, get_head_pose
from utils.torch_utils import select_device
from utils.general import check_img_size, scale_coords, non_max_suppression
from utils.datasets import LoadImages
from models.experimental import attempt_load

def plot_3axis_Zaxis(img, yaw, pitch, roll, tdx=None, tdy=None, size=50., limited=True, thickness=2):
    p = pitch * np.pi / 180
    y = -(yaw * np.pi / 180)
    r = roll * np.pi / 180
    if tdx != None and tdy != None:
        face_x = tdx
        face_y = tdy
    else:
        height, width = img.shape[:2]
        face_x = width / 2
        face_y = height / 2

    x1 = size * (math.cos(y) * math.cos(r)) + face_x
    y1 = size * (math.cos(p) * math.sin(r) + math.cos(r) * math.sin(p) * math.sin(y)) + face_y
    x2 = size * (-math.cos(y) * math.sin(r)) + face_x
    y2 = size * (math.cos(p) * math.cos(r) - math.sin(p) * math.sin(y) * math.sin(r)) + face_y
    x3 = size * (math.sin(y)) + face_x
    y3 = size * (-math.cos(y) * math.sin(p)) + face_y

    scale_ratio = 2
    base_len = math.sqrt((face_x - x3)**2 + (face_y - y3)**2)
    if face_x == x3:
        endx = tdx
        if face_y < y3:
            if limited:
                endy = tdy + (y3 - face_y) * scale_ratio
            else:
                endy = img.shape[0]
        else:
            if limited:
                endy = tdy - (face_y - y3) * scale_ratio
            else:
                endy = 0
    elif face_x > x3:
        if limited:
            endx = tdx - (face_x - x3) * scale_ratio
            endy = tdy - (face_y - y3) * scale_ratio
        else:
            endx = 0
            endy = tdy - (face_y - y3) / (face_x - x3) * tdx
    else:
        if limited:
            endx = tdx + (x3 - face_x) * scale_ratio
            endy = tdy + (y3 - face_y) * scale_ratio
        else:
            endx = img.shape[1]
            endy = tdy - (face_y - y3) / (face_x - x3) * (tdx - endx)

    cv2.line(img, (int(tdx), int(tdy)), (int(endx), int(endy)), (0,255,255), thickness)
    cv2.line(img, (int(face_x), int(face_y)), (int(x1),int(y1)),(0,0,255),thickness)
    cv2.line(img, (int(face_x), int(face_y)), (int(x2),int(y2)),(0,255,0),thickness)
    cv2.line(img, (int(face_x), int(face_y)), (int(x3),int(y3)),(255,0,0),thickness)
    return img

def plot_vectors(img, gaze_vector, point_vector, face_center_x, face_center_y, x, y, scale=50):
    gaze_x = int(gaze_vector[0] * scale + face_center_x)
    gaze_y = int(gaze_vector[1] * scale + face_center_y)
    point_x = int(point_vector[0] * scale + face_center_x)
    point_y = int(point_vector[1] * scale + face_center_y)

    cv2.line(img, (int(face_center_x), int(face_center_y)), (gaze_x, gaze_y), (255, 0, 255), 2)
    cv2.line(img, (int(x), int(y)), (int(face_center_x), int(face_center_y)), (255, 255, 0), 2)
    return img

def is_point_in_gaze_direction(face_box, pitch, yaw, roll, x, y, threshold=90):
    face_center_x = (face_box[0] + face_box[2]) / 2
    face_center_y = (face_box[1] + face_box[3]) / 2

    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(pitch)
    roll_rad = math.radians(roll)

    Rx = np.array([[1, 0, 0],
                   [0, math.cos(pitch_rad), -math.sin(pitch_rad)],
                   [0, math.sin(pitch_rad), math.cos(pitch_rad)]])

    Ry = np.array([[math.cos(yaw_rad), 0, math.sin(yaw_rad)],
                   [0, 1, 0],
                   [-math.sin(yaw_rad), 0, math.cos(yaw_rad)]])

    Rz = np.array([[math.cos(roll_rad), -math.sin(roll_rad), 0],
                   [math.sin(roll_rad), math.cos(roll_rad), 0],
                   [0, 0, 1]])

    R = np.dot(Rz, np.dot(Ry, Rx))

    gaze_vector = np.array([0, 0, -1])
    gaze_vector = np.dot(R, gaze_vector)

    point_vector = np.array([x - face_center_x, y - face_center_y, 0])
    point_vector = point_vector / np.linalg.norm(point_vector)
    dot_product = np.dot(gaze_vector, point_vector)
    angle = math.degrees(math.acos(dot_product))

    if angle <= threshold:
        return True, gaze_vector[:2], point_vector[:2]
    else:
        return False, gaze_vector[:2], point_vector[:2]

def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x_overlap = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
    y_overlap = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
    overlap_area = x_overlap * y_overlap

    area_box1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area_box2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area_box1 + area_box2 - overlap_area

    iou = overlap_area / union_area
    return iou

def match_boxes(head_info, watcher_box, opener_box):
    assert len(head_info) == 2
 
    head_info1, head_info2 = head_info
    head_pose_1, head_box1 = head_info1
    head_pose_2, head_box2 = head_info2

    iou1_watcher = calculate_iou(head_box1, watcher_box)
    iou2_watcher = calculate_iou(head_box2, watcher_box)
    iou1_opener = calculate_iou(head_box1, opener_box)
    iou2_opener = calculate_iou(head_box2, opener_box)

    if max(iou1_watcher, iou2_opener) >= max(iou2_watcher, iou1_opener):
        return (head_info1, watcher_box), (head_info2, opener_box)
    else:
        return (head_info2, watcher_box), (head_info1, opener_box)

def check(pose_info, head_info):
    human1 = pose_info[0]
    human2 = pose_info[1]

    human1_keypoints = human1['keypoints']
    human2_keypoints = human2['keypoints']

    opener, watcher = None, None
    if human1_keypoints[9][0] > human2_keypoints[9][0]: #左手手腕最靠近画面右侧的是开门的
        opener = human1
        watcher = human2
    else:
        opener = human2
        watcher = human1
    
    box_pair = match_boxes(head_info, watcher['bbox'][0], opener['bbox'][0])
    watcher_head_info = box_pair[0][0]
    watcher_head_pose, watcher_head_box = watcher_head_info
    pitch, yaw, roll = watcher_head_pose
    x, y = opener['keypoints'][9]
    return is_point_in_gaze_direction(watcher_head_box, pitch, yaw, roll, x, y)[0]

def visualize_frame(frame, pose_info, head_info):
    human1 = pose_info[0]
    human2 = pose_info[1]

    human1_keypoints = human1['keypoints']
    human2_keypoints = human2['keypoints']

    opener, watcher = None, None
    if human1_keypoints[9][0] > human2_keypoints[9][0]:
        opener = human1
        watcher = human2
    else:
        opener = human2
        watcher = human1
    
    box_pair = match_boxes(head_info, watcher['bbox'][0], opener['bbox'][0])
    watcher_head_info = box_pair[0][0]
    watcher_head_pose, watcher_head_box = watcher_head_info
    pitch, yaw, roll = watcher_head_pose
    x, y = opener['keypoints'][9]

    face_center_x = (watcher_head_box[0] + watcher_head_box[2]) / 2
    face_center_y = (watcher_head_box[1] + watcher_head_box[3]) / 2

    #frame = plot_3axis_Zaxis(frame, yaw, pitch, roll, face_center_x, face_center_y)

    result, gaze_vector, point_vector = is_point_in_gaze_direction(watcher_head_box, pitch, yaw, roll, x, y)
    frame = plot_vectors(frame, gaze_vector, point_vector, face_center_x, face_center_y, x, y)

    return frame

def get_video_acc(video_path, inferencer):


    mp4name = video_path.split('/')[-1]
    output_video = f'output_video/{mp4name}'
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    valid_frame_count = 0
    acc = 0
    for i in range(total_frames):
        #pose_info = human_pose[i]['predictions'][0]
        ret, frame = video.read()
        pose_info = next(inferencer(frame))['predictions'][0]
        if len(pose_info) != 2:
            out.write(frame)
            continue
        _, head_info = get_head_pose(frame)
        if len(head_info) != 2:
            out.write(frame)
            continue
        valid_frame_count += 1
        if check(pose_info, head_info):
            acc += 1
            text = 'Right'
        else:    
            text = 'Wrong'
        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(text, font, 1, 2)[0]
        
        textX = (frame.shape[1] - textsize[0]) // 2 
        textY = (frame.shape[0] + textsize[1]) // 2
        cv2.putText(frame, text, (textX, textY), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        frame = visualize_frame(frame, pose_info, head_info)
        out.write(frame)
        #cv2.imwrite(f"temp/{i}.jpg", frame)
    print('valid_frame_count', valid_frame_count)
    print('total_frame_count', total_frames)
    if valid_frame_count == 0:
        return -1 #没有双人有效帧
    return acc / valid_frame_count

if __name__ == "__main__":

    inferencer = MMPoseInferencer('td-hm_hrnet-w32_8xb64-210e_coco-256x192')
    video_path = 'open_door/wrong-3-29/20240329_160630.mp4'
    print(get_video_acc(video_path, inferencer))
    #print(old_get_video_acc(video_path))

     