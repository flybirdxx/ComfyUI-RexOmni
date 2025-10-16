"""
Rex-Omni ComfyUI 节点工具函数
包含图像转换、数据处理、可视化等工具函数
"""

import os
import json
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Optional, Tuple, Any

# 尝试导入 ComfyUI 的路径管理器
try:
    import folder_paths
    COMFYUI_AVAILABLE = True
except ImportError:
    # 如果不在 ComfyUI 环境中，使用默认路径
    COMFYUI_AVAILABLE = False

# 模型路径设置
REX_OMNI_MODEL_DIR = "Rex-Omni"
current_dir = os.path.dirname(os.path.abspath(__file__))

if COMFYUI_AVAILABLE:
    # 使用 ComfyUI 的模型目录 - 从 custom_nodes 目录向上两级到 ComfyUI 根目录
    MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
else:
    # 默认模型目录
    MODELS_DIR = os.path.join(current_dir, "models")

# 直接使用 ComfyUI 的 models 目录
model_dir = os.path.join(MODELS_DIR, REX_OMNI_MODEL_DIR)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 直接使用 Rex-Omni 模型目录
model_list = ["Rex-Omni"]  # 直接使用目录名


def tensor_to_pil(tensor):
    """将 ComfyUI 的 IMAGE (Tensor) 转换为 PIL Image"""
    if tensor is None:
        return None
    
    try:
        
        # ComfyUI的IMAGE格式是 (B, H, W, C)
        if tensor.dim() == 4:  # (B, H, W, C)
            tensor = tensor[0]  # 取第一个批次，形状变为 (H, W, C)
        elif tensor.dim() == 3:  # (H, W, C)
            pass  # 已经是正确的格式
        else:
            return None
        
        # 转换为numpy数组
        image_np = tensor.cpu().numpy()
        
        # 确保值在[0,1]范围内
        image_np = np.clip(image_np, 0, 1)
        
        # 转换为0-255范围
        image_np = (image_np * 255).astype(np.uint8)
        
        # 创建PIL图像
        pil_image = Image.fromarray(image_np)
        return pil_image
        
    except Exception as e:
        return None


def pil_to_tensor(image):
    """将 PIL Image 转换为 ComfyUI 的 IMAGE (Tensor)"""
    if image is None:
        return None
    
    try:
        # 确保图像是RGB格式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 转换为numpy数组，形状为 (H, W, C)
        image_array = np.array(image).astype(np.float32) / 255.0
        
        
        # 转换为tensor，形状为 (H, W, C)
        tensor = torch.from_numpy(image_array)
        
        # 添加批次维度，形状变为 (1, H, W, C) - 这是ComfyUI的标准格式
        tensor = tensor.unsqueeze(0)
        
        return tensor
        
    except Exception as e:
        return None


def format_predictions_json(result):
    """格式化预测结果为JSON字符串"""
    try:
        if not result:
            return "{}"
        
        # 检查结果是否为字典
        if not isinstance(result, dict):
            return "{}"
        
        # 根据实际的推理结果格式处理
        if "extracted_predictions" in result:
            predictions = result["extracted_predictions"]
            return json.dumps(predictions, ensure_ascii=False, indent=2)
        else:
            # 如果没有extracted_predictions，返回空字典
            return "{}"
            
    except Exception as e:
        return "{}"


def extract_bboxes(result):
    """提取边界框数据 - 输出KJNodes兼容格式"""
    try:
        if not result or not isinstance(result, dict):
            return []
        
        bboxes = []
        
        # 处理extracted_predictions格式
        if "extracted_predictions" in result:
            predictions = result["extracted_predictions"]
            for category, objects in predictions.items():
                for obj in objects:
                    if "coords" in obj and obj.get("type") == "box":
                        coords = obj["coords"]
                        # 转换为KJNodes期望的格式 [x_min, y_min, width, height]
                        if len(coords) >= 4:
                            x_min, y_min, x_max, y_max = coords[:4]
                            width = x_max - x_min
                            height = y_max - y_min
                            bbox = [x_min, y_min, width, height]
                            bboxes.append(bbox)
        
        # 处理直接格式的边界框
        elif "bboxes" in result:
            for bbox in result["bboxes"]:
                if isinstance(bbox, dict) and "coords" in bbox:
                    coords = bbox["coords"]
                    if len(coords) >= 4:
                        x_min, y_min, x_max, y_max = coords[:4]
                        width = x_max - x_min
                        height = y_max - y_min
                        bbox = [x_min, y_min, width, height]
                        bboxes.append(bbox)
        
        # 处理原始预测格式
        elif "predictions" in result:
            for pred in result["predictions"]:
                if "bbox" in pred:
                    coords = pred["bbox"]
                    if len(coords) >= 4:
                        x_min, y_min, x_max, y_max = coords[:4]
                        width = x_max - x_min
                        height = y_max - y_min
                        bbox = [x_min, y_min, width, height]
                        bboxes.append(bbox)
        
        return bboxes
    except Exception as e:
        return []


def extract_texts(result):
    """提取文本数据 - 增强OCR支持"""
    try:
        if not result or not isinstance(result, dict):
            return "{}"
        
        texts = []
        
        # 处理extracted_predictions格式
        if "extracted_predictions" in result:
            predictions = result["extracted_predictions"]
            for category, objects in predictions.items():
                for obj in objects:
                    if "text" in obj:
                        text_data = {
                            "category": category,
                            "text": obj["text"],
                            "coords": obj.get("coords", []),
                            "confidence": obj.get("confidence", 0.0)
                        }
                        texts.append(text_data)
        
        # 处理OCR专用格式
        elif "texts" in result:
            for text in result["texts"]:
                if isinstance(text, dict):
                    texts.append(text)
                else:
                    # 简单文本格式
                    texts.append({
                        "category": "text",
                        "text": str(text),
                        "coords": [],
                        "confidence": 1.0
                    })
        
        # 处理原始预测格式
        elif "predictions" in result:
            for pred in result["predictions"]:
                if "text" in pred:
                    text_data = {
                        "category": pred.get("category", "text"),
                        "text": pred["text"],
                        "coords": pred.get("coords", []),
                        "confidence": pred.get("confidence", 0.0)
                    }
                    texts.append(text_data)
        
        return json.dumps(texts, ensure_ascii=False, indent=2)
    except Exception as e:
        return "{}"


def extract_keypoints(result):
    """提取关键点数据 - 增强姿态估计支持"""
    try:
        if not result or not isinstance(result, dict):
            return "{}"
        
        keypoints = []
        
        # 处理extracted_predictions格式
        if "extracted_predictions" in result:
            predictions = result["extracted_predictions"]
            for category, objects in predictions.items():
                for obj in objects:
                    if "keypoints" in obj:
                        kp_data = {
                            "category": category,
                            "keypoints": obj["keypoints"],
                            "confidence": obj.get("confidence", 0.0)
                        }
                        keypoints.append(kp_data)
        
        # 处理关键点专用格式
        elif "keypoints" in result:
            for kp in result["keypoints"]:
                if isinstance(kp, dict):
                    keypoints.append(kp)
                else:
                    # 简单关键点格式
                    keypoints.append({
                        "category": "person",
                        "keypoints": kp,
                        "confidence": 1.0
                    })
        
        # 处理原始预测格式
        elif "predictions" in result:
            for pred in result["predictions"]:
                if "keypoints" in pred:
                    kp_data = {
                        "category": pred.get("category", "person"),
                        "keypoints": pred["keypoints"],
                        "confidence": pred.get("confidence", 0.0)
                    }
                    keypoints.append(kp_data)
        
        return json.dumps(keypoints, ensure_ascii=False, indent=2)
    except Exception as e:
        return "{}"


def format_result(result, task):
    """格式化检测结果"""
    if not result:
        return "未检测到目标"
    
    # 检查结果是否为字典
    if not isinstance(result, dict):
        return f"结果格式: {type(result)}"
    
    # 根据实际的推理结果格式处理
    if "extracted_predictions" in result:
        predictions = result["extracted_predictions"]
        if not predictions:
            return "未检测到目标"
        
        # 统计各类别的数量
        category_counts = {}
        total_objects = 0
        
        for category, objects in predictions.items():
            count = len(objects)
            category_counts[category] = count
            total_objects += count
        
        # 构建结果文本
        result_text = f"检测到 {total_objects} 个目标:\n"
        for category, count in category_counts.items():
            result_text += f"  - {category}: {count} 个\n"
        
        return result_text
    else:
        return f"结果格式: {list(result.keys())}"


def create_visualization(image, result, task):
    """创建可视化图像 - 使用官方RexOmniVisualize"""
    try:
        if not result:
            return image
        
        # 支持多种数据格式
        predictions = None
        
        # 格式1: extracted_predictions
        if isinstance(result, dict) and "extracted_predictions" in result:
            predictions = result["extracted_predictions"]
        # 格式2: 直接是predictions
        elif isinstance(result, dict) and any(key for key in result.keys() if isinstance(result[key], list)):
            predictions = result
        # 格式3: 其他可能的格式
        elif isinstance(result, dict):
            # 尝试找到包含检测结果的键
            for key, value in result.items():
                if isinstance(value, dict) and any(isinstance(v, list) for v in value.values()):
                    predictions = value
                    break
        
        if predictions:
            
            # 导入RexOmniVisualize
            try:
                from rex_omni.utils import RexOmniVisualize
            except ImportError:
                return image
            
            # 对于keypoint任务，直接使用原始数据格式（与官方示例一致）
            if task == "keypoint":
                
                vis_image = RexOmniVisualize(
                    image=image,
                    predictions=predictions,  # 直接使用原始数据
                    font_size=20,  # 增加字体大小以提高可读性
                    draw_width=2,  # 使用官方设置
                    show_labels=True  # 显示标签，与官方示例一致
                )
            else:
                # 对于其他任务，进行数据格式转换
                formatted_predictions = convert_to_visualization_format(predictions, task)
                
                vis_image = RexOmniVisualize(
                    image=image,
                    predictions=formatted_predictions,
                    font_size=20,  # 增加字体大小以提高可读性
                    draw_width=2,  # 使用官方设置
                    show_labels=True  # 显示标签，与官方示例一致
                )
            
            return vis_image
        else:
            # 如果没有找到预测数据，返回原图像
            return image
        
    except Exception as e:
        return image


def convert_to_visualization_format(predictions, task="detection"):
    """将预测数据转换为RexOmniVisualize期望的格式"""
    try:
        
        if not predictions:
            return {}
        
        # 如果已经是正确的格式，直接返回
        if isinstance(predictions, dict) and all(isinstance(v, list) for v in predictions.values()):
            return predictions
        
        # 转换格式
        formatted = {}
        
        if task == "detection":
            # 检测任务：将边界框数据转换为标准格式
            for category, objects in predictions.items():
                formatted[category] = []
                for obj in objects:
                    if "coords" in obj and obj.get("type") == "box":
                        formatted_obj = {
                            "type": "box",
                            "coords": obj["coords"],
                            "confidence": obj.get("confidence", 0.0)
                        }
                        formatted[category].append(formatted_obj)
        
        elif task == "keypoint":
            # 关键点任务：保持原始格式
            formatted = predictions
        
        elif task == "ocr":
            # OCR任务：将文本数据转换为标准格式
            for category, objects in predictions.items():
                formatted[category] = []
                for obj in objects:
                    if "text" in obj:
                        formatted_obj = {
                            "type": "text",
                            "text": obj["text"],
                            "coords": obj.get("coords", []),
                            "confidence": obj.get("confidence", 0.0)
                        }
                        formatted[category].append(formatted_obj)
        
        else:
            # 其他任务：保持原始格式
            formatted = predictions
        
        return formatted
        
    except Exception as e:
        return predictions


def load_font(font_size: int) -> ImageFont.ImageFont:
    """加载字体"""
    try:
        # 尝试加载系统字体
        try:
            return ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
            except:
                return ImageFont.load_default()
    except:
        return ImageFont.load_default()


def draw_box(
    draw: ImageDraw.ImageDraw,
    coords: List[float],
    color: str = "red",
    width: int = 2,
    label: str = "",
    font: Optional[ImageFont.ImageFont] = None
):
    """绘制边界框"""
    if len(coords) < 4:
        return
    
    x1, y1, x2, y2 = coords[:4]
    
    # 绘制边界框
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    
    # 绘制标签
    if label and font:
        # 计算文本位置
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # 标签背景
        label_y = y1 - text_height - 4
        if label_y < 0:
            label_y = y1 + 4
        
        draw.rectangle(
            [x1, label_y, x1 + text_width + 4, label_y + text_height + 4],
            fill=color
        )
        
        # 标签文本
        draw.text((x1 + 2, label_y + 2), label, fill="white", font=font)


def draw_point(
    draw: ImageDraw.ImageDraw,
    coords: List[float],
    color: str = "red",
    radius: int = 3,
    label: str = "",
    font: Optional[ImageFont.ImageFont] = None
):
    """绘制点"""
    if len(coords) < 2:
        return
    
    x, y = coords[:2]
    
    # 绘制点
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)
    
    # 绘制标签
    if label and font:
        # 计算文本位置
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # 标签背景
        label_x = x + radius + 2
        label_y = y - text_height // 2
        
        draw.rectangle(
            [label_x, label_y, label_x + text_width + 4, label_y + text_height + 4],
            fill=color
        )
        
        # 标签文本
        draw.text((label_x + 2, label_y + 2), label, fill="white", font=font)


def draw_keypoints(draw, annotation, color, draw_width, category, font, show_labels):
    """绘制关键点"""
    try:
        keypoints = annotation.get('keypoints', {})
        if not keypoints:
            return
        
        # 绘制关键点
        for kp_name, kp_coords in keypoints.items():
            if kp_coords == "unvisible":
                continue
            
            if isinstance(kp_coords, list) and len(kp_coords) == 2:
                x, y = kp_coords
                
                # 绘制关键点
                draw.ellipse(
                    [x - draw_width, y - draw_width, x + draw_width, y + draw_width],
                    fill=color
                )
                
                # 绘制标签
                if show_labels and font:
                    label = f"{category}_{kp_name}"
                    text_bbox = draw.textbbox((0, 0), label, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    
                    # 标签背景
                    label_x = x + draw_width + 2
                    label_y = y - text_height // 2
                    
                    draw.rectangle(
                        [label_x, label_y, label_x + text_width + 4, label_y + text_height + 4],
                        fill=color
                    )
                    
                    # 标签文本
                    draw.text((label_x + 2, label_y + 2), label, fill="white", font=font)
        
        # 绘制骨架连接（如果有关键点）
        if len(keypoints) > 1:
            draw_skeleton(draw, keypoints, color)
            
    except Exception as e:
        pass


def draw_skeleton(draw, keypoints, color):
    """绘制骨架连接"""
    try:
        # 定义骨架连接关系（以人体关键点为例）
        skeleton_connections = [
            # 头部连接
            ("nose", "left_eye"),
            ("nose", "right_eye"),
            ("left_eye", "left_ear"),
            ("right_eye", "right_ear"),
            # 躯干连接
            ("nose", "neck"),
            ("neck", "left_shoulder"),
            ("neck", "right_shoulder"),
            ("left_shoulder", "right_shoulder"),
            # 手臂连接
            ("left_shoulder", "left_elbow"),
            ("left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow"),
            ("right_elbow", "right_wrist"),
            # 躯干连接
            ("left_shoulder", "left_hip"),
            ("right_shoulder", "right_hip"),
            ("left_hip", "right_hip"),
            # 腿部连接
            ("left_hip", "left_knee"),
            ("left_knee", "left_ankle"),
            ("right_hip", "right_knee"),
            ("right_knee", "right_ankle"),
        ]
        
        # 绘制连接线
        for start_kp, end_kp in skeleton_connections:
            if start_kp in keypoints and end_kp in keypoints:
                start_coords = keypoints[start_kp]
                end_coords = keypoints[end_kp]
                
                # 检查坐标是否有效
                if (isinstance(start_coords, list) and len(start_coords) == 2 and
                    isinstance(end_coords, list) and len(end_coords) == 2 and
                    start_coords != "unvisible" and end_coords != "unvisible"):
                    
                    draw.line([start_coords[0], start_coords[1], end_coords[0], end_coords[1]], 
                             fill=color, width=2)
                    
    except Exception as e:
        pass


def get_model_directory():
    """获取模型目录"""
    return model_dir


def get_model_list():
    """获取模型列表"""
    return model_list


def check_environment():
    """检查环境信息"""
    return {
        "comfyui_available": COMFYUI_AVAILABLE,
        "model_dir": model_dir,
        "model_list": model_list
    }
