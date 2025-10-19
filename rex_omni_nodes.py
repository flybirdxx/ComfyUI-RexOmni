import os
import sys
import json
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Optional, Tuple, Any

# 添加 src 目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

# 导入工具函数
from .utils import (
    get_model_directory, get_model_list, check_environment,
    tensor_to_pil, pil_to_tensor,
    create_visualization, format_result, format_predictions_json,
    convert_to_visualization_format, extract_bboxes, extract_texts, extract_keypoints,
    load_font, draw_box, draw_point
)

try:
    from rex_omni.wrapper import RexOmniWrapper
    from rex_omni.tasks import TaskType
    from rex_omni.utils import RexOmniVisualize
    REX_OMNI_AVAILABLE = True
except ImportError as e:
    REX_OMNI_AVAILABLE = False
    # 定义默认的TaskType以防导入失败
    from enum import Enum
    class TaskType(Enum):
        DETECTION = "detection"
        POINTING = "pointing"
        VISUAL_PROMPTING = "visual_prompting"
        KEYPOINT = "keypoint"
        OCR_BOX = "ocr_box"
        OCR_POLYGON = "ocr_polygon"
        GUI_DETECTION = "gui_grounding"
        GUI_POINTING = "gui_pointing"

# 获取模型配置
model_dir = get_model_directory()
model_list = get_model_list()
env_info = check_environment()


class RexOmniLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (model_list,),
                "backend": (["transformers", "vllm"], {"default": "transformers"}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("REX_OMNI_MODEL",)
    RETURN_NAMES = ("rex_omni_model",)
    FUNCTION = "load_model"
    CATEGORY = "Rex-Omni"

    def __init__(self):
        self.model = None
        self.model_name = None

    def load_model(self, model_name, backend, max_tokens, temperature):
        if not REX_OMNI_AVAILABLE:
            raise RuntimeError("Rex-Omni 模块未正确安装")
        
        # 检查是否已经加载了相同的模型
        if self.model is not None and self.model_name == model_name:
            return (self.model,)
        
        try:
            model_path = model_dir
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型路径不存在: {model_path}")
            
            self.model = RexOmniWrapper(
                model_path=model_path,
                backend=backend,
                max_tokens=max_tokens,
                temperature=temperature
            )
            self.model_name = model_name
            return (self.model,)
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {str(e)}")

class RexOmniDetector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rex_omni_model": ("REX_OMNI_MODEL",),
                "image": ("IMAGE",),
                "task": ([task.value for task in TaskType], {"default": "detection"}),
                "text_prompt": ("STRING", {"multiline": True, "default": "a person"}),
            },
            "optional": {
                "keypoint_type": (["person", "animal"], {"default": "person"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "BBOX", "STRING", "STRING")
    RETURN_NAMES = ("visualization", "result_text", "predictions_json", "bboxes", "texts", "keypoints")
    FUNCTION = "detect"
    CATEGORY = "Rex-Omni"

    def detect(self, rex_omni_model, image, task, text_prompt, keypoint_type="human"):
        if not REX_OMNI_AVAILABLE:
            raise RuntimeError("Rex-Omni 模块未正确安装")
        
        try:
            pil_image = tensor_to_pil(image)
            if pil_image is None:
                raise ValueError("无法处理输入图像")
            
            inference_params = {
                "images": pil_image,
                "task": task,
            }
            
            if task == "keypoint":
                inference_params["keypoint_type"] = keypoint_type
                if keypoint_type == "animal":
                    if text_prompt and text_prompt.strip():
                        inference_params["categories"] = [text_prompt.strip()]
                    else:
                        inference_params["categories"] = ["animal"]
                else:
                    inference_params["categories"] = ["person"]
            else:
                inference_params["categories"] = text_prompt
            
            result = rex_omni_model.inference(**inference_params)
            
            if result:
                if isinstance(result, dict):
                    processed_result = result
                elif isinstance(result, list) and len(result) > 0:
                    processed_result = result[0]
                else:
                    processed_result = result
                
                visualization = create_visualization(pil_image, processed_result, task)
            else:
                processed_result = None
                visualization = pil_image
            
            vis_tensor = pil_to_tensor(visualization)
            result_text = format_result(processed_result, task)
            predictions_json = format_predictions_json(processed_result)
            bboxes_data = extract_bboxes(processed_result)
            texts_data = extract_texts(processed_result)
            keypoints_data = extract_keypoints(processed_result)
            return (vis_tensor, result_text, predictions_json, bboxes_data, texts_data, keypoints_data)
            
        except Exception as e:
            return (image, f"检测失败: {str(e)}", "{}", "{}", "{}", "{}", "{}")


class ColorGenerator:
    def __init__(self, color_type: str = "text"):
        self.color_type = color_type
        
        if color_type == "same":
            self.color = tuple((np.random.randint(0, 127, size=3) + 128).tolist())
        elif color_type == "text":
            np.random.seed(3396)
            self.num_colors = 300
            self.colors = np.random.randint(0, 127, size=(self.num_colors, 3)) + 128
        else:
            raise ValueError(f"未知颜色类型: {color_type}")
    
    def get_color(self, text: str) -> Tuple[int, int, int]:
        if self.color_type == "same":
            return self.color
        
        if self.color_type == "text":
            text_hash = hash(text)
            index = text_hash % self.num_colors
            color = tuple(self.colors[index])
            return color
        
        raise ValueError(f"未知颜色类型: {self.color_type}")


def _draw_keypoints(draw, annotation, color, draw_width, category, font, show_labels):
    try:
        keypoints = annotation.get("keypoints", {})
        
        if not keypoints:
            return
        
        skeleton_connections = [
            ("left eye", "right eye"),
            ("left eye", "nose"),
            ("right eye", "nose"),
            ("nose", "left ear"),
            ("nose", "right ear"),
            ("left ear", "left shoulder"),
            ("right ear", "right shoulder"),
            ("left shoulder", "right shoulder"),
            ("left shoulder", "left elbow"),
            ("right shoulder", "right elbow"),
            ("left elbow", "left wrist"),
            ("right elbow", "right wrist"),
            ("left shoulder", "left hip"),
            ("right shoulder", "right hip"),
            ("left hip", "right hip"),
            ("left hip", "left knee"),
            ("right hip", "right knee"),
            ("left knee", "left ankle"),
            ("right knee", "right ankle"),
        ]
        
        hand_skeleton_connections = [
            ("wrist", "thumb root"),
            ("thumb root", "thumb's third knuckle"),
            ("thumb's third knuckle", "thumb's second knuckle"),
            ("thumb's second knuckle", "thumb's first knuckle"),
            ("wrist", "forefinger's root"),
            ("forefinger's root", "forefinger's third knuckle"),
            ("forefinger's third knuckle", "forefinger's second knuckle"),
            ("forefinger's second knuckle", "forefinger's first knuckle"),
            ("wrist", "middle finger's root"),
            ("middle finger's root", "middle finger's third knuckle"),
            ("middle finger's third knuckle", "middle finger's second knuckle"),
            ("middle finger's second knuckle", "middle finger's first knuckle"),
            ("wrist", "ring finger's root"),
            ("ring finger's root", "ring finger's third knuckle"),
            ("ring finger's third knuckle", "ring finger's second knuckle"),
            ("ring finger's second knuckle", "ring finger's first knuckle"),
            ("wrist", "pinky finger's root"),
            ("pinky finger's root", "pinky finger's third knuckle"),
            ("pinky finger's third knuckle", "pinky finger's second knuckle"),
            ("pinky finger's second knuckle", "pinky finger's first knuckle"),
        ]
        
        animal_skeleton_connections = [
            ("left eye", "right eye"),
            ("left eye", "nose"),
            ("right eye", "nose"),
            ("nose", "neck"),
            ("neck", "left shoulder"),
            ("neck", "right shoulder"),
            ("left shoulder", "left elbow"),
            ("right shoulder", "right elbow"),
            ("left elbow", "left front paw"),
            ("right elbow", "right front paw"),
            ("neck", "left hip"),
            ("neck", "right hip"),
            ("left hip", "left knee"),
            ("right hip", "right knee"),
            ("left knee", "left back paw"),
            ("right knee", "right back paw"),
            ("neck", "root of tail"),
        ]
        
        if "wrist" in keypoints:
            connections = hand_skeleton_connections
        elif "left shoulder" in keypoints and "left hip" in keypoints:
            connections = skeleton_connections
        else:
            connections = animal_skeleton_connections
        
        try:
            from PIL import ImageFont
            font = ImageFont.load_default()
        except:
            font = None
        
        for point_name, coords in keypoints.items():
            if isinstance(coords, list) and len(coords) >= 2:
                x, y = coords[0], coords[1]
                radius = 3
                draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color, outline="white", width=1)
                
                if show_labels and font:
                    label_x, label_y = x + 8, y - 8
                    try:
                        bbox = draw.textbbox((label_x, label_y), point_name, font)
                        box_h = bbox[3] - bbox[1]
                        box_w = bbox[2] - bbox[0]
                        padding = 2
                        
                        draw.rectangle([
                            label_x - padding,
                            label_y - box_h - padding,
                            label_x + box_w + padding,
                            label_y + padding,
                        ], fill=color)
                        
                        draw.text((label_x, label_y - box_h), point_name, fill="white", font=font)
                    except:
                        pass
        
        for start_point, end_point in connections:
            if start_point in keypoints and end_point in keypoints:
                start_coords = keypoints[start_point]
                end_coords = keypoints[end_point]
                
                if (isinstance(start_coords, list) and len(start_coords) >= 2 and
                    isinstance(end_coords, list) and len(end_coords) >= 2):
                    
                    start_x, start_y = start_coords[0], start_coords[1]
                    end_x, end_y = end_coords[0], end_coords[1]
                    
                    draw.line([start_x, start_y, end_x, end_y], fill=color, width=draw_width)
    except Exception as e:
        pass


def RexOmniVisualize(
    image: Image.Image,
    predictions: Dict[str, List[Dict]],
    font_size: int = 15,
    draw_width: int = 6,
    show_labels: bool = True,
    custom_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
) -> Image.Image:
    vis_image = image.copy()
    draw = ImageDraw.Draw(vis_image)
    
    font = load_font(font_size)
    color_generator = ColorGenerator("text")
    
    for category, annotations in predictions.items():
        if custom_colors and category in custom_colors:
            color = color_generator.get_color(category)
        else:
            color = color_generator.get_color(category)
        
        for i, annotation in enumerate(annotations):
            annotation_type = annotation.get("type", "box")
            
            if annotation_type == "keypoint":
                _draw_keypoints(draw, annotation, color, draw_width, category, font, show_labels)
            else:
                coords = annotation.get("coords", [])
                
                if annotation_type == "box" and len(coords) == 4:
                    draw_box(draw, coords, color, draw_width, category, font, show_labels)
                elif annotation_type == "point" and len(coords) == 2:
                    draw_point(
                        draw, coords, color, draw_width, category, font, show_labels
                    )
    
    return vis_image


NODE_CLASS_MAPPINGS = {
    "RexOmniLoader": RexOmniLoader,
    "RexOmniDetector": RexOmniDetector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RexOmniLoader": "Rex-Omni Loader",
    "RexOmniDetector": "Rex-Omni Detector",
}
