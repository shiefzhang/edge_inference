import logging
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
from ultralytics import YOLO

from utils import draw_detections
from utils import DetectionResult
from utils import box_colors
from detect_human import detect_human

logger = logging.getLogger(__name__)

def unvest(image: Image.Image, 
          conf_threshold: float = 0.25,
          human_model: YOLO = None,
          unvest_cls_model: YOLO = None) -> Tuple[Image.Image, List[DetectionResult]]:
    """
    检测图片中未穿戴反光衣的人员
    
    Args:
        image: PIL Image对象
        conf_threshold: 置信度阈值
        
    Returns:
        检测结果列表
    """
    # global unvest_cls_model, vest1_idx
    vest1_idx = 2

    detections = detect_human(image, conf_threshold=conf_threshold, human_model=human_model)
    logger.info(f"检测到 {len(detections)} 个目标:")
    
    if not detections:
        logger.info("  [无检测结果]")
        return detections

    for human in detections:
        if human.type == "human":
            human_box = human.box if hasattr(human, "box") else []
            # 截取识别出的人员图像
            human_img = image.crop(human_box)
            # 使用反光衣分类模型进行预测，判断是否穿戴反光衣
            vest_results = unvest_cls_model.predict(np.array(human_img), conf=0.25, verbose=False)

            # 获取第一个结果
            result = vest_results[0] if vest_results else None

            is_vest = False
            if result and hasattr(result, 'probs'):
                # 获取最高概率类别和置信度
                probs = result.probs
                
                # 获取最高概率的索引和值
                if hasattr(probs, 'top1') and hasattr(probs, 'top1conf'):
                    top_class_idx = probs.top1
                    top_conf = probs.top1conf
                    
                    logger.info(f"最高概率类别索引: {top_class_idx}, 置信度: {top_conf}")
                    
                    if top_class_idx in [vest1_idx] and top_conf >= 0.2:
                        is_vest = True  # 穿戴了反光衣
                    else:
                        is_vest = False  # 未穿戴反光衣
                else:
                    is_vest = False  # 无法确定，认为未穿戴
            else:
                is_vest = False  # 没有检测结果，认为未穿戴

            if not is_vest: # 如果没有检测到穿戴反光衣，认为存在违规
                 human.violation = True
                 human.viol_content = f"人员未检测到反光衣，存在安全风险"
                 human.viol_color = list(box_colors["alarm"])
                 logger.info(f"  [违规] 人员({human.track_id}) 未检测到反光衣，存在安全风险")
            else:
                 human.violation = False
                 human.viol_color = list(box_colors.get(human.type, box_colors["unknown"]))

    # 绘制检测结果（可选）
    img_array = np.array(image)
    infer_img = draw_detections(
        img_array,
        [d.model_dump() if hasattr(d, "model_dump") else d for d in detections] # 将DetectionResult对象转换为字典，以便draw_detections使用
    )

    return infer_img, detections
