import numpy as np
from PIL import Image
from typing import List, Optional
from pydantic import BaseModel, Field
from itertools import zip_longest
from ultralytics import YOLO

from utils import DetectionResult
from utils import box_colors
from utils import normalize_box

def detect_default_person(image: Image.Image, conf_threshold: float = 0.25,default_model: YOLO = None) -> List[DetectionResult]:
    """
    使用YOLO模型检测默认人员场景
    
    Args:
        image: PIL Image对象
        conf_threshold: 置信度阈值
        
    Returns:
        检测结果列表
    """
    # global default_model
    # global default_person_idx
    default_person_idx = 0  # 默认人员类别索引，需根据实际模型调整
    
    if default_model is None:
        raise RuntimeError("默认模型未加载")
    
    # 转换为numpy数组
    img_array = np.array(image)
    img_h, img_w = img_array.shape[:2]
    
    # 执行默认人员场景检测
    person_results = default_model.track(
        img_array,
        persist=True,
        classes=[default_person_idx],
        conf=conf_threshold,
        verbose=False
    )
    person_ids, person_boxes, person_confs = [], [], []
    if person_results and person_results[0].boxes:
        person_ids = person_results[0].boxes.id.int().cpu().tolist() if person_results[0].boxes.id is not None else []
        person_boxes = person_results[0].boxes.xyxy.cpu().numpy()
        person_confs = person_results[0].boxes.conf.cpu().numpy()

    detections = []
    if len(person_boxes) > 0:
        # 如果检测到工作人员，记录信息
        for p_id, p_conf, p_xyxy in zip_longest(person_ids, person_confs, person_boxes, fillvalue=None):
            person_box = list(map(int, p_xyxy))
            person_area = (person_box[2] - person_box[0]) * (person_box[3] - person_box[1])
            person_label = f"ID:{p_id}({p_conf:.2f})"
            detection = DetectionResult(
                type="person",
                track_id=p_id,
                label=person_label,
                box=person_box,
                norm_box=normalize_box(person_box, img_w, img_h),
                area=person_area,
                conf=p_conf,
                violation=None,  # 违规状态需要后续判断
                viol_content=None,  # 违规内容需要后续判断
                viol_color=list(box_colors["person"])
            )
            detections.append(detection)

    return detections
