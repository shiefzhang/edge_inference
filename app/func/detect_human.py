import logging
import numpy as np
from PIL import Image
from typing import List, Optional
from pydantic import BaseModel, Field
from itertools import zip_longest
import torch
from ultralytics import YOLO

from utils import DetectionResult
from utils import box_colors
from utils import normalize_box

logger = logging.getLogger(__name__)

def detect_human(image: Image.Image, conf_threshold: float = 0.25,human_model: YOLO = None) -> List[DetectionResult]:
    """
    使用YOLO模型检测人员违规场景
    
    Args:
        image: PIL Image对象
        conf_threshold: 置信度阈值
        
    Returns:
        检测结果列表
    """
    # global human_model
    # global human_idx, unhat_idx, unvest_idx, phone_idx, smoke_idx
    # global human_unclear_idx, vest_idx, hat_idx
    human_idx = 4
    unhat_idx = 3
    unvest_idx = 1
    hat_idx = 0
    vest_idx = 2
    human_unclear_idx = 7
    phone_idx = 5
    smoke_idx = 6

    if human_model is None:
        raise RuntimeError("人员违规场景模型未加载")
    
    # 转换为numpy数组
    img_array = np.array(image)
    img_h, img_w = img_array.shape[:2]
    
    # 执行人员违规场景检测
    human_results = human_model.track(
        img_array,
        persist=True,
        # classes=[human_idx, unhat_idx, unvest_idx, phone_idx, smoke_idx, human_unclear_idx, vest_idx, hat_idx],
        classes=[human_idx, human_unclear_idx],
        conf=conf_threshold,
        verbose=False
    )
    # 获取第一个图像的结果
    result = human_results[0]

    # 根据类别索引分开
    human_indices = result.boxes.cls == human_idx
    unhat_indices = result.boxes.cls == unhat_idx
    unvest_indices = result.boxes.cls == unvest_idx
    phone_indices = result.boxes.cls == phone_idx
    smoke_indices = result.boxes.cls == smoke_idx

    human_unclear_indices = result.boxes.cls == human_unclear_idx
    vest_indices = result.boxes.cls == vest_idx
    hat_indices = result.boxes.cls == hat_idx

    # 获取各自的检测框、ID和置信度
    human_boxes = result.boxes[human_indices].xyxy.cpu().numpy() if human_indices.any() else []
    unhat_boxes = result.boxes[unhat_indices].xyxy.cpu().numpy() if unhat_indices.any() else []
    unvest_boxes = result.boxes[unvest_indices].xyxy.cpu().numpy() if unvest_indices.any() else []
    phone_boxes = result.boxes[phone_indices].xyxy.cpu().numpy() if phone_indices.any() else []
    smoke_boxes = result.boxes[smoke_indices].xyxy.cpu().numpy() if smoke_indices.any() else []

    human_unclear_boxes = result.boxes[human_unclear_indices].xyxy.cpu().numpy() if human_unclear_indices.any() else []
    vest_boxes = result.boxes[vest_indices].xyxy.cpu().numpy() if vest_indices.any() else []
    hat_boxes = result.boxes[hat_indices].xyxy.cpu().numpy() if hat_indices.any() else []
    
    human_ids = result.boxes[human_indices].id.int().cpu().tolist() if human_indices.any() and result.boxes[human_indices].id is not None else []
    unhat_ids = result.boxes[unhat_indices].id.int().cpu().tolist() if unhat_indices.any() and result.boxes[unhat_indices].id is not None else []
    unvest_ids = result.boxes[unvest_indices].id.int().cpu().tolist() if unvest_indices.any() and result.boxes[unvest_indices].id is not None else []
    phone_ids = result.boxes[phone_indices].id.int().cpu().tolist() if phone_indices.any() and result.boxes[phone_indices].id is not None else []
    smoke_ids = result.boxes[smoke_indices].id.int().cpu().tolist() if smoke_indices.any() and result.boxes[smoke_indices].id is not None else []

    human_unclear_ids = result.boxes[human_unclear_indices].id.int().cpu().tolist() if human_unclear_indices.any() and result.boxes[human_unclear_indices].id is not None else []
    vest_ids = result.boxes[vest_indices].id.int().cpu().tolist() if vest_indices.any() and result.boxes[vest_indices].id is not None else []
    hat_ids = result.boxes[hat_indices].id.int().cpu().tolist() if hat_indices.any() and result.boxes[hat_indices].id is not None else []

    human_confs = result.boxes[human_indices].conf.cpu().numpy() if human_indices.any() else []
    unhat_confs = result.boxes[unhat_indices].conf.cpu().numpy() if unhat_indices.any() else []
    unvest_confs = result.boxes[unvest_indices].conf.cpu().numpy() if unvest_indices.any() else []
    phone_confs = result.boxes[phone_indices].conf.cpu().numpy() if phone_indices.any() else []
    smoke_confs = result.boxes[smoke_indices].conf.cpu().numpy() if smoke_indices.any() else []
    
    human_unclear_confs = result.boxes[human_unclear_indices].conf.cpu().numpy() if human_unclear_indices.any() else []
    vest_confs = result.boxes[vest_indices].conf.cpu().numpy() if vest_indices.any() else []
    hat_confs = result.boxes[hat_indices].conf.cpu().numpy() if hat_indices.any() else []

    detections = []
    if len(human_boxes) > 0:
        # 如果检测到人员违规场景，记录信息
        for t_id, t_conf, t_xyxy in zip_longest(human_ids, human_confs, human_boxes, fillvalue=None):
            human_box = list(map(int, t_xyxy))
            human_area = (human_box[2] - human_box[0]) * (human_box[3] - human_box[1])
            human_label = f"ID:{t_id}({t_conf:.2f})"
            detection = DetectionResult(
                type="human",
                track_id=t_id,
                label=human_label,
                box=human_box,
                norm_box=normalize_box(human_box, img_w, img_h),
                area=human_area,
                conf=t_conf,
                violation=None,  # 违规状态需要后续判断
                viol_content=None,  # 违规内容需要后续判断
                viol_color=list(box_colors["human"])
            )
            detections.append(detection)


    if len(unhat_boxes) > 0:
        # 如果检测到未佩戴安全帽，记录信息
        for t_id, t_conf, t_xyxy in zip_longest(unhat_ids, unhat_confs, unhat_boxes, fillvalue=None):
            unhat_box = list(map(int, t_xyxy))
            unhat_area = (unhat_box[2] - unhat_box[0]) * (unhat_box[3] - unhat_box[1])
            unhat_label = f"ID:{t_id}({t_conf:.2f})"
            detection = DetectionResult(
                type="unhat",
                track_id=t_id,
                label=unhat_label,
                box=unhat_box,
                norm_box=normalize_box(unhat_box, img_w, img_h),
                area=unhat_area,
                conf=t_conf,
                violation=True,
                viol_content="未佩戴安全帽，存在安全风险",
                viol_color=list(box_colors["alarm"])
            )
            logger.info(f"  [违规] 人员({t_id}) 未佩戴安全帽，存在安全风险 (置信度:{t_conf:.2f})")
            detections.append(detection)

    if len(unvest_boxes) > 0:
        # 如果检测到未穿戴反光衣，记录信息
        for t_id, t_conf, t_xyxy in zip_longest(unvest_ids, unvest_confs, unvest_boxes, fillvalue=None):
            unvest_box = list(map(int, t_xyxy))
            unvest_area = (unvest_box[2] - unvest_box[0]) * (unvest_box[3] - unvest_box[1])
            unvest_label = f"ID:{t_id}({t_conf:.2f})"
            detection = DetectionResult(
                type="unvest",
                track_id=t_id,
                label=unvest_label,
                box=unvest_box,
                norm_box=normalize_box(unvest_box, img_w, img_h),
                area=unvest_area,
                conf=t_conf,
                violation=True,
                viol_content="未穿戴反光衣，存在安全风险",
                viol_color=list(box_colors["alarm"])
            )
            logger.info(f"  [违规] 人员({t_id}) 未穿戴反光衣，存在安全风险 (置信度:{t_conf:.2f})")
            detections.append(detection)

    if len(phone_boxes) > 0:
        # 如果检测到使用手机，记录信息
        for t_id, t_conf, t_xyxy in zip_longest(phone_ids, phone_confs, phone_boxes, fillvalue=None):
            phone_box = list(map(int, t_xyxy))
            phone_area = (phone_box[2] - phone_box[0]) * (phone_box[3] - phone_box[1])
            phone_label = f"ID:{t_id}({t_conf:.2f})"
            detection = DetectionResult(
                type="phone",
                track_id=t_id,
                label=phone_label,
                box=phone_box,
                norm_box=normalize_box(phone_box, img_w, img_h),
                area=phone_area,
                conf=t_conf,
                violation=True,
                viol_content="人员使用手机，存在安全风险",
                viol_color=list(box_colors["alarm"])
            )
            logger.info(f"  [违规] 人员({t_id}) 使用手机，存在安全风险 (置信度:{t_conf:.2f})")
            detections.append(detection)

    if len(smoke_boxes) > 0:
        # 如果检测到吸烟，记录信息
        for t_id, t_conf, t_xyxy in zip_longest(smoke_ids, smoke_confs, smoke_boxes, fillvalue=None):
            smoke_box = list(map(int, t_xyxy))
            smoke_area = (smoke_box[2] - smoke_box[0]) * (smoke_box[3] - smoke_box[1])
            smoke_label = f"ID:{t_id}({t_conf:.2f})"
            detection = DetectionResult(
                type="smoke",
                track_id=t_id,
                label=smoke_label,
                box=smoke_box,
                norm_box=normalize_box(smoke_box, img_w, img_h),
                area=smoke_area,
                conf=t_conf,
                violation=True,
                viol_content="人员吸烟，存在安全风险",
                viol_color=list(box_colors["alarm"])
            )
            logger.info(f"  [违规] 人员({t_id}) 吸烟，存在安全风险 (置信度:{t_conf:.2f})")
            detections.append(detection)

    if len(human_unclear_boxes) > 0:
        # 如果检测到人员但不清晰，记录信息
        for t_id, t_conf, t_xyxy in zip_longest(human_unclear_ids, human_unclear_confs, human_unclear_boxes, fillvalue=None):
            human_unclear_box = list(map(int, t_xyxy))
            human_unclear_area = (human_unclear_box[2] - human_unclear_box[0]) * (human_unclear_box[3] - human_unclear_box[1])
            human_unclear_label = f"ID:{t_id}({t_conf:.2f})"
            detection = DetectionResult(
                type="human_unclear",
                track_id=t_id,
                label=human_unclear_label,
                box=human_unclear_box,
                norm_box=normalize_box(human_unclear_box, img_w, img_h),
                area=human_unclear_area,
                conf=t_conf,
                violation=True,
                viol_content="误识别为人员，存在安全风险",
                viol_color=list(box_colors["alarm"])
            )
            logger.info(f"  [违规] 人员({t_id}) 误识别为人员，存在安全风险 (置信度:{t_conf:.2f})")
            detections.append(detection)
    
    if len(vest_boxes) > 0:
        # 如果检测到穿戴反光衣，记录信息
        for t_id, t_conf, t_xyxy in zip_longest(vest_ids, vest_confs, vest_boxes, fillvalue=None):
            vest_box = list(map(int, t_xyxy))
            vest_area = (vest_box[2] - vest_box[0]) * (vest_box[3] - vest_box[1])
            vest_label = f"ID:{t_id}({t_conf:.2f})"
            detection = DetectionResult(
                type="vest",
                track_id=t_id,
                label=vest_label,
                box=vest_box,
                norm_box=normalize_box(vest_box, img_w, img_h),
                area=vest_area,
                conf=t_conf,
                violation=False,
                viol_content="穿戴反光衣，符合安全要求",
                viol_color=list(box_colors["vest"])
            )
            logger.info(f"  [正常] 人员({t_id}) 穿戴反光衣，符合安全要求 (置信度:{t_conf:.2f})")
            detections.append(detection)
    
    if len(hat_boxes) > 0:
        # 如果检测到佩戴安全帽，记录信息
        for t_id, t_conf, t_xyxy in zip_longest(hat_ids, hat_confs, hat_boxes, fillvalue=None):
            hat_box = list(map(int, t_xyxy))
            hat_area = (hat_box[2] - hat_box[0]) * (hat_box[3] - hat_box[1])
            hat_label = f"ID:{t_id}({t_conf:.2f})"
            detection = DetectionResult(
                type="hat",
                track_id=t_id,
                label=hat_label,
                box=hat_box,
                norm_box=normalize_box(hat_box, img_w, img_h),
                area=hat_area,
                conf=t_conf,
                violation=False,
                viol_content="佩戴安全帽，符合安全要求",
                viol_color=list(box_colors["hat"])
            )
            logger.info(f"  [正常] 人员({t_id}) 佩戴安全帽，符合安全要求 (置信度:{t_conf:.2f})")
            detections.append(detection)

    return detections


