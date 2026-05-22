import cv2
import logging
import numpy as np
from PIL import Image
from PIL import ImageFont, ImageDraw
from pathlib import Path
try:
    from shapely.geometry import LineString, Polygon, MultiLineString
    from shapely.ops import unary_union
except ImportError:
    LineString = Polygon = MultiLineString = unary_union = None

from typing import List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)



class DetectionResult(BaseModel):
    """单个检测结果"""
    type: str = Field(..., description="检测类型")
    track_id: Optional[int] = Field(None, description="跟踪ID")
    label: str = Field(..., description="检测标签")
    box: List[int] = Field(..., description="边界框坐标 [x1, y1, x2, y2]")
    norm_box: Optional[List[float]] = Field(None, description="归一化边界框坐标 [x1/w, y1/h, x2/w, y2/h]")
    mask: Optional[List[List[int]]] = Field(None, description="分割掩码的多边形坐标列表")
    contour: Optional[List[List[int]]] = Field(None, description="轮廓坐标列表")
    area: Optional[int] = Field(None, description="边界框面积")
    conf: Optional[float] = Field(None, description="置信度")
    violation: Optional[bool] = Field(None, description="作业是否违规")
    viol_content: Optional[str] = Field(None, description="违规内容描述")
    viol_color: Optional[List[int]] = Field(None, description="违规标识颜色，RGB格式")
    lifting: Optional[bool] = Field(None, description="起重机是否正在作业")
    body_id: Optional[int] = Field(None, description="机体跟踪ID")
    body_lbl: Optional[str] = Field(None, description="机体检测标签")
    body_box: Optional[List[int]] = Field(None, description="机体边界框坐标 [x1, y1, x2, y2]")
    body_area: Optional[int] = Field(None, description="机体边界框面积")
    body_pos: Optional[List[int]] = Field(None, description="机体底部中心点坐标 [x, y]")
    class Config:
        populate_by_name = True

box_colors = {
    # RGB colors. Keep this in sync with PIL/numpy image arrays.
    "alarm": (255, 0, 0),           # 红色-警报/违规
    "safe": (0, 0, 255),            # 蓝色-安全
    "crane": (0, 255, 0),           # 绿色-起重机
    "body": (128, 128, 0),          # 橄榄色-本体
    "person": (0, 255, 255),        # 青色-人员
    "hook": (0, 165, 255),          # 天蓝色-吊钩
    "craneload": (128, 0, 128),     # 紫色-吊载
    "cranecargo": (255, 0, 0),      # 红色-吊货
    "pit": (255, 255, 0),           # 黄色-基坑
    "edge_prot": (255, 0, 255),     # 粉色-边界防护
    "ladder": (0, 128, 255),        # 蓝色-梯子
    "equip": (128, 0, 128),         # 紫色-机械设备
    "unknown": (255, 255, 255),     # 白色-未知

    "vest": (128, 255, 0),          # 黄绿色-反光衣
    "hat": (0, 255, 128),           # 青绿色-安全帽
    "human": (0, 128, 255),         # 蓝色-人员
    # "unhat": (128, 0, 255),       # 紫色-未戴安全帽
    "human_unclear": (192, 192, 192) # 灰色-识别不清

}

def normalize_box(box, width, height):
    if not box or len(box) != 4 or width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = box
    return [
        max(0.0, min(1.0, x1 / width)),
        max(0.0, min(1.0, y1 / height)),
        max(0.0, min(1.0, x2 / width)),
        max(0.0, min(1.0, y2 / height)),
    ]

def load_chinese_font(font_size):
    font_paths = [
        str(Path(__file__).resolve().parent / "SIMHEI.TTF"),
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simsun.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]

    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, font_size)
        except OSError:
            pass
    logger.info("警告：未找到可用中文字体，可能仍显示乱码")
    return ImageFont.load_default()

def get_text_size(text, font_size):
    font = load_chinese_font(font_size)
    if hasattr(font, "getbbox"):
        left, top, right, bottom = font.getbbox(text)
        return right - left, bottom - top
    return font.getsize(text)

def cv2_add_chinese_text(img, text, position, font_size=20, color=(0, 0, 0)):
    """
    给RGB图像添加中文字符（解决问号问题）
    Args:
        img: RGB格式的图像
        text: 要绘制的中文字符串
        position: 文字左上角坐标 (x, y)
        font_size: 字体大小（默认20）
        color: 文字颜色（RGB格式，默认黑色）
    Returns:
        添加文字后的RGB图像
    """
    pil_img = Image.fromarray(img)
    font = load_chinese_font(font_size)
    
    # 3. 绘制文字
    draw = ImageDraw.Draw(pil_img)
    draw.text(position, text, font=font, fill=color)
    
    return np.array(pil_img)

def get_contrast_color(rgb_color, threshold=127):
    """
    根据传入的RGB颜色，返回对比色（黑色/白色）
    逻辑：浅色背景返回黑色，深色背景返回白色
    
    Args:
        rgb_color: RGB格式的颜色元组 (R, G, B)，如 (0,255,0)、(255,0,0)
        threshold: 亮度阈值（0-255，默认127，>127为浅色，<127为深色）
    
    Returns:
        对比色元组 (B, G, R)，要么是黑色(0,0,0)，要么是白色(255,255,255)
    """
    # 1. 提取RGB通道值
    r, g, b = rgb_color
    
    # 2. 计算人眼感知的亮度（更符合视觉体验，而非简单平均）
    # 公式：L = 0.299R + 0.587G + 0.114B（国际标准亮度计算）
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    
    # 3. 根据亮度阈值返回对比色
    if brightness > threshold:
        return (0, 0, 0)   # 浅色背景 → 黑色字体
    else:
        return (255, 255, 255)  # 深色背景 → 白色字体

def draw_detections(
    img,
    detections: list,
    font_scale: float = 0.8,
    thickness: int = 2
):
    """
    在图片上绘制检测结果
    
    Args:
        img: RGB图片数组
        detections: 检测结果列表，每个元素包含 'box', 'label', 'track_id', 'type', 'conf' 等字段
    """
    for det in detections:
        det_type = det.get("type", "unknown")
        color = box_colors.get(det_type, (255, 255, 255))
        font_size = int(font_scale * 30)  # 按原font_scale换算字体大小（可调整）
        if det_type == "crane":
            body_color = box_colors["body"]
        box = det.get("box", [])
        label = det.get("label", "unknown")
        violation = det.get("violation", False)
        if violation:
            color = box_colors.get("alarm", (255, 255, 255))  # 红色-违规
            viol_content = det.get("viol_content", "")
            if viol_content:
                label += f" [{viol_content}]"
        det["viol_color"] = list(color)
        font_color = get_contrast_color(color)

        # 基坑特殊处理：使用轮廓绘制而非矩形框
        if det_type == "pit":
            contour_list = det.get("contour", [])
            if not contour_list:
                continue
            # 1. 将列表转换回OpenCV需要的格式
            contour_array = np.array(contour_list, dtype=np.int32)  # 形状: (N, 2)
            contour_for_cv2 = contour_array.reshape(-1, 1, 2)  # 形状: (N, 1, 2) - OpenCV标准格式

            # 创建半透明填充层
            overlay = img.copy()
            cv2.fillPoly(overlay, [contour_for_cv2], color)
            # 设置透明度并合并
            alpha = 0.3  # 透明度
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
            # img[:] = result

            # 绘制轮廓线
            cv2.drawContours(img, [contour_for_cv2], -1, color, 2)


            if LineString is None or Polygon is None or MultiLineString is None or unary_union is None:
                continue

            # 1. 将轮廓点转换为LineString（轮廓线）
            contour_line = LineString(contour_list)

            # 2. 将所有切割矩形合并为一个几何对象
            cut_polygons = []
            for edge in detections:
                if edge.get("type") == "edge_prot":
                    edge_box = edge.get("box", [])
                    if len(edge_box) == 4:
                        ex1, ey1, ex2, ey2 = edge_box
                        rect_poly = Polygon([
                            [ex1, ey1],
                            [ex2, ey1],
                            [ex2, ey2],
                            [ex1, ey2]
                        ])
                        cut_polygons.append(rect_poly)

            # 合并所有切割区域
            cut_union = unary_union(cut_polygons)
            
            # 3. 从轮廓线中减去矩形部分
            uncut_lines = contour_line.difference(cut_union.buffer(0.1))  # 微小缓冲确保相交
            
            # 4. 处理结果（可能得到多个线段）
            if uncut_lines.is_empty:
                logger.info("轮廓被完全切割，无剩余部分")
            # 5. 在图像上绘制未被切割的部分
            elif isinstance(uncut_lines, LineString):
                # 单个线段
                lines = [uncut_lines]
            elif isinstance(uncut_lines, MultiLineString):
                # 多个线段
                lines = list(uncut_lines.geoms)
            else:
                lines = []

            for line in lines:
                if not line.is_empty:
                    # 获取线段顶点
                    coords = list(line.coords)
                    if len(coords) > 1:  # 确保是线段
                        # 转换为OpenCV格式
                        points = np.array(coords, dtype=np.int32)
                        points = points.reshape((-1, 1, 2))
                        length = line.length
                        if length > 100:  # 过滤掉过短的线段
                            # 加粗绘制未被切割的轮廓
                            cv2.polylines(img, [points], False, (255, 0, 0), 1) # 红色加粗线表示未被切割的部分




        # 其他类型使用矩形框绘制
        else:
            if len(box) == 4:
                x1, y1, x2, y2 = box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
                # 添加标签背景
                text_w, text_h = get_text_size(label, font_size)
                pad_x, pad_y = 4, 3
                label_top = y1 - text_h - pad_y * 2
                label_bottom = y1
                if label_top < 0:
                    label_top = y2
                    label_bottom = y2 + text_h + pad_y * 2
                cv2.rectangle(img, (x1, label_top), (x1 + text_w + pad_x * 2, label_bottom), color, -1)
                img = cv2_add_chinese_text(img, label, (x1 + pad_x, label_top + pad_y), font_size, font_color)
        
        # 如果是起重机，绘制本体边界框和标签
        if det_type == "crane":
            body_box = det.get("body_box", [])
            body_label = det.get("body_lbl", "unknown")
            if body_box and len(body_box) == 4:
                bx1, by1, bx2, by2 = body_box
                cv2.rectangle(img, (bx1, by1), (bx2, by2), body_color, thickness)
                
                # 添加标签背景
                text_w, text_h = get_text_size(body_label, font_size)
                pad_x, pad_y = 4, 3
                label_top = by1 - text_h - pad_y * 2
                label_bottom = by1
                if label_top < 0:
                    label_top = by2
                    label_bottom = by2 + text_h + pad_y * 2
                cv2.rectangle(img, (bx1, label_top), (bx1 + text_w + pad_x * 2, label_bottom), body_color, -1)
                img = cv2_add_chinese_text(img, body_label, (bx1 + pad_x, label_top + pad_y), font_size, font_color)
    
    return img
