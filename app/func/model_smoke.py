import logging
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
from ultralytics import YOLO

from utils import draw_detections
from utils import DetectionResult
from utils import box_colors
from detect_person import detect_default_person

logger = logging.getLogger(__name__)

def smoking(image: Image.Image, 
          conf_threshold: float = 0.25,
          default_model: YOLO = None,
          smoke_cls_model: YOLO = None) -> Tuple[Image.Image, List[DetectionResult]]:
    """
    检测图片中人员吸烟的情况
    
    Args:
        image: PIL Image对象
        conf_threshold: 置信度阈值
        
    Returns:
        检测结果列表
    """
    # global smoke_cls_model, smoke1_idx
    smoke1_idx = 1  # 吸烟类别索引，需根据实际模型调整

    detections = detect_default_person(image, conf_threshold=conf_threshold, default_model=default_model)
    logger.debug(f"检测到 {len(detections)} 个目标:")
    
    if not detections:
        logger.debug("  [无检测结果]")
        return detections

    for person in detections:
        if person.type == "person":
            person_box = person.box if hasattr(person, "box") else []
            # 截取识别出的人员图像
            person_img = image.crop(person_box)
            # 使用吸烟检测分类模型进行预测，判断是否吸烟
            smoking_results = smoke_cls_model.predict(np.array(person_img), conf=0.25, verbose=False)
            # 获取第一个结果
            result = smoking_results[0] if smoking_results else None

            is_smoking = False
            if result and hasattr(result, 'probs'):
                # 获取最高概率类别和置信度
                probs = result.probs
                
                # 获取最高概率的索引和值
                if hasattr(probs, 'top1') and hasattr(probs, 'top1conf'):
                    top_class_idx = probs.top1
                    top_conf = probs.top1conf
                    
                    logger.debug(f"最高概率类别索引: {top_class_idx}, 置信度: {top_conf}")
                    
                    if top_class_idx in [smoke1_idx] and top_conf >= 0.2:
                        is_smoking = True  # 吸烟了
                    else:
                        is_smoking = False  # 未吸烟
                else:
                    is_smoking = False  # 无法确定，认为未吸烟
            else:
                is_smoking = False  # 没有检测结果，认为未吸烟

            if is_smoking: # 如果检测到吸烟，认为存在违规
                 person.violation = True
                 person.viol_content = f"人员吸烟，存在安全风险"
                 person.viol_color = list(box_colors["alarm"])
                 logger.info(f"  [违规] 人员({person.track_id}) 吸烟，存在安全风险")
            else:
                 person.violation = False
                 person.viol_color = list(box_colors.get(person.type, box_colors["unknown"]))

    # 绘制检测结果（可选）
    img_array = np.array(image)
    infer_img = draw_detections(
        img_array,
        [d.model_dump() if hasattr(d, "model_dump") else d for d in detections] # 将DetectionResult对象转换为字典，以便draw_detections使用
    )

    return infer_img, detections
