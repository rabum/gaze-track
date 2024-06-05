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
from gaze import YOLOv8_face
from gaze import L2CSNet
from utils.mmpose import inference_video, calculate_face_bbox, get_head_pose, plot_3axis_Zaxis
from utils.torch_utils import select_device
from utils.general import check_img_size, scale_coords, non_max_suppression
from utils.datasets import LoadImages
from models.experimental import attempt_load

def visualize_frame(frame, pose_info, head_info):
    human1 = pose_info[0]
    human2 = pose_info[1]

    human1_keypoints = human1['keypoints']
    human2_keypoints = human2['keypoints']

    opener, watcher = None, None
    if human1_keypoints[10][0] > human2_keypoints[10][0]:
        opener = human1
        watcher = human2
    else:
        opener = human2
        watcher = human1
    
    box_pair = match_boxes(head_info, watcher['bbox'][0], opener['bbox'][0])
    watcher_head_info = box_pair[0][0]
    watcher_head_pose, watcher_head_box = watcher_head_info
    pitch, yaw, roll = watcher_head_pose
    x, y = opener['keypoints'][10]

    face_center_x = (watcher_head_box[0] + watcher_head_box[2]) / 2
    face_center_y = (watcher_head_box[1] + watcher_head_box[3]) / 2

    frame = plot_3axis_Zaxis(frame, yaw, pitch, roll, face_center_x, face_center_y)

    result, gaze_vector, point_vector = is_point_in_gaze_direction(watcher_head_box, pitch, yaw, roll, x, y)
    frame = plot_vectors(frame, gaze_vector, point_vector, face_center_x, face_center_y)

    return frame


def plot_vectors(img, gaze_vector, point_vector, face_center_x, face_center_y, scale=50):
    gaze_x = int(gaze_vector[0] * scale + face_center_x)
    gaze_y = int(gaze_vector[1] * scale + face_center_y)
    point_x = int(point_vector[0] * scale + face_center_x)
    point_y = int(point_vector[1] * scale + face_center_y)

    cv2.line(img, (int(face_center_x), int(face_center_y)), (gaze_x, gaze_y), (255, 0, 255), 2)
    cv2.line(img, (int(face_center_x), int(face_center_y)), (point_x, point_y), (255, 255, 0), 2)
    return img


def is_point_in_gaze_direction(face_box, pitch, yaw, roll, x, y, threshold=15):
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


    print('angle', angle)
    if angle <= threshold:
        return True, 
    else:
        return False


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
    if human1_keypoints[10][0] > human2_keypoints[10][0]: # 看谁的右手离画面右侧更近
        opener = human1
        watcher = human2
    else:
        opener = human2
        watcher = human1
    
    box_pair = match_boxes(head_info, watcher['bbox'][0], opener['bbox'][0])
    watcher_head_info = box_pair[0][0]
    watcher_head_pose, watcher_head_box = watcher_head_info
    pitch, yaw, roll = watcher_head_pose
    x, y = opener['keypoints'][10]
    return is_point_in_gaze_direction(watcher_head_box, pitch, yaw, roll, x, y)


if __name__ == "__main__":
    
    video_path = 'open_door/right-3-26/0c86149835824a67b260879276328105.mp4'
    #video_path = 'open_door/passing_by-3-26/1d8a98793b00403f16d224770c3fc269.mp4'
    human_pose = inference_video(video_path)
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    acc = 0
    for i in range(len(human_pose)):
        pose_info = human_pose[i]['predictions'][0]
        if len(pose_info) <= 1: #人体姿态识别只检测到了一个人
            continue
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = video.read()
        _, head_info = get_head_pose(frame)
        if len(head_info) <= 1: #头部姿态识别只检测到了一个人
            continue
        if check(pose_info, head_info):
            acc += 1
    print(acc / total_frames)
        

        
    