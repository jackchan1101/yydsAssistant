#!/usr/bin/env python3
"""
自适应阈值识别系统
根据图像特征自动调整预处理参数，提高识别准确率
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ImageType(Enum):
    """图像类型枚举"""
    UI_ELEMENT = "ui_element"      # UI元素（按钮、图标）
    TEXT = "text"                  # 文本区域
    COMPLEX_BACKGROUND = "complex"  # 复杂背景
    LOW_CONTRAST = "low_contrast"  # 低对比度
    UNKNOWN = "unknown"            # 未知类型

@dataclass
class ImageAnalysisResult:
    """图像分析结果"""
    image_type: ImageType
    brightness: float              # 亮度 (0-255)
    contrast: float               # 对比度
    noise_level: float           # 噪声水平
    edge_density: float          # 边缘密度
    recommended_params: Dict[str, Any]  # 推荐参数

class AdaptiveThresholdSystem:
    """自适应阈值识别系统"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化自适应阈值系统
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        self._init_default_config()
        
        # 参数历史记录（用于自适应调整）
        self.parameter_history: List[Dict] = []
        
        logger.info("自适应阈值识别系统初始化完成")
    
    def _init_default_config(self):
        """初始化默认配置"""
        default_config = {
            'analysis': {
                'brightness_threshold_low': 50,
                'brightness_threshold_high': 200,
                'contrast_threshold_low': 20,
                'edge_density_threshold': 0.05,
            },
            'preprocessing': {
                # 不同图像类型的预处理参数
                'ui_element': {
                    'grayscale': True,
                    'denoise': True,
                    'threshold_type': 'adaptive',
                    'blur_kernel': (3, 3),
                },
                'text': {
                    'grayscale': True,
                    'denoise': True,
                    'threshold_type': 'otsu',
                    'scale_factor': 2.0,
                },
                'complex_background': {
                    'grayscale': True,
                    'denoise': True,
                    'threshold_type': 'adaptive',
                    'blur_kernel': (5, 5),
                    'morphology': True,
                },
                'low_contrast': {
                    'grayscale': True,
                    'contrast_enhance': True,
                    'clahe_clip_limit': 2.0,
                    'threshold_type': 'adaptive',
                }
            }
        }
        
        # 合并配置
        for key, value in default_config.items():
            if key not in self.config:
                self.config[key] = value
    
    def analyze_image(self, image: np.ndarray) -> ImageAnalysisResult:
        """
        分析图像特征
        
        Args:
            image: 输入图像
            
        Returns:
            ImageAnalysisResult: 分析结果
        """
        # 转换为灰度图（如果必要）
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 计算图像特征
        brightness = self._calculate_brightness(gray)
        contrast = self._calculate_contrast(gray)
        noise_level = self._estimate_noise(gray)
        edge_density = self._calculate_edge_density(gray)
        
        # 判断图像类型
        image_type = self._classify_image_type(brightness, contrast, noise_level, edge_density)
        
        # 生成推荐参数
        recommended_params = self._generate_recommended_params(image_type)
        
        result = ImageAnalysisResult(
            image_type=image_type,
            brightness=brightness,
            contrast=contrast,
            noise_level=noise_level,
            edge_density=edge_density,
            recommended_params=recommended_params
        )
        
        logger.debug(f"图像分析完成: 类型={image_type.value}, "
                    f"亮度={brightness:.1f}, 对比度={contrast:.1f}, "
                    f"噪声={noise_level:.3f}, 边缘密度={edge_density:.3f}")
        
        return result
    
    def _calculate_brightness(self, image: np.ndarray) -> float:
        """计算图像亮度"""
        return np.mean(image)
    
    def _calculate_contrast(self, image: np.ndarray) -> float:
        """计算图像对比度"""
        return np.std(image)
    
    def _estimate_noise(self, image: np.ndarray) -> float:
        """估计图像噪声水平"""
        # 使用拉普拉斯算子的方差估计噪声
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        # 归一化到0-1范围
        return min(laplacian_var / 1000, 1.0)
    
    def _calculate_edge_density(self, image: np.ndarray) -> float:
        """计算边缘密度"""
        # 使用Canny边缘检测
        edges = cv2.Canny(image, 50, 150)
        # 计算边缘像素比例
        edge_pixels = np.sum(edges > 0)
        total_pixels = image.shape[0] * image.shape[1]
        return edge_pixels / total_pixels
    
    def _classify_image_type(self, brightness: float, contrast: float, 
                           noise: float, edge_density: float) -> ImageType:
        """根据特征分类图像类型"""
        analysis_config = self.config.get('analysis', {})
        
        # 判断低对比度
        if contrast < analysis_config.get('contrast_threshold_low', 20):
            return ImageType.LOW_CONTRAST
        
        # 判断文本区域（高边缘密度）
        if edge_density > analysis_config.get('edge_density_threshold', 0.05):
            return ImageType.TEXT
        
        # 判断复杂背景（高噪声）
        if noise > 0.1:
            return ImageType.COMPLEX_BACKGROUND
        
        # 默认认为是UI元素
        return ImageType.UI_ELEMENT
    
    def _generate_recommended_params(self, image_type: ImageType) -> Dict[str, Any]:
        """根据图像类型生成推荐参数"""
        preprocessing_config = self.config.get('preprocessing', {})
        base_params = preprocessing_config.get(image_type.value, {})
        
        # 返回副本，避免修改原始配置
        return base_params.copy()
    
    def adaptive_preprocess(self, image: np.ndarray, 
                          custom_params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        自适应图像预处理
        
        Args:
            image: 输入图像
            custom_params: 自定义参数（可选）
            
        Returns:
            np.ndarray: 预处理后的图像
        """
        # 分析图像特征
        analysis_result = self.analyze_image(image)
        
        # 合并参数（自定义参数优先）
        params = analysis_result.recommended_params.copy()
        if custom_params:
            params.update(custom_params)
        
        # 记录参数使用
        self.parameter_history.append({
            'image_type': analysis_result.image_type.value,
            'params': params,
            'timestamp': np.datetime64('now')
        })
        
        # 执行预处理
        processed = self._apply_preprocessing(image, params)
        
        logger.info(f"自适应预处理完成: 类型={analysis_result.image_type.value}, "
                   f"参数={params}")
        
        return processed
    
    def _apply_preprocessing(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """应用预处理参数"""
        processed = image.copy()
        
        # 转换为灰度图
        if params.get('grayscale', True) and len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        
        # 对比度增强
        if params.get('contrast_enhance', False):
            processed = self._enhance_contrast(processed)
        
        # 去噪
        if params.get('denoise', False):
            blur_kernel = params.get('blur_kernel', (3, 3))
            if blur_kernel[0] > 0 and blur_kernel[1] > 0:
                processed = cv2.GaussianBlur(processed, blur_kernel, 0)
        
        # CLAHE对比度限制自适应直方图均衡化
        if params.get('clahe', False):
            clahe = cv2.createCLAHE(
                clipLimit=params.get('clahe_clip_limit', 2.0),
                tileGridSize=params.get('clahe_grid_size', (8, 8))
            )
            processed = clahe.apply(processed)
        
        # 阈值处理
        threshold_type = params.get('threshold_type')
        if threshold_type == 'otsu':
            _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif threshold_type == 'adaptive':
            processed = cv2.adaptiveThreshold(
                processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
        elif threshold_type == 'binary':
            threshold_value = params.get('threshold_value', 127)
            _, processed = cv2.threshold(processed, threshold_value, 255, cv2.THRESH_BINARY)
        
        # 形态学操作
        if params.get('morphology', False):
            kernel = np.ones((2, 2), np.uint8)
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
        
        # 缩放（用于OCR）
        scale_factor = params.get('scale_factor', 1.0)
        if scale_factor != 1.0:
            new_width = int(processed.shape[1] * scale_factor)
            new_height = int(processed.shape[0] * scale_factor)
            processed = cv2.resize(processed, (new_width, new_height), 
                                 interpolation=cv2.INTER_CUBIC)
        
        return processed
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """增强图像对比度"""
        # 转换为YUV色彩空间
        if len(image.shape) == 3:
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            y, u, v = cv2.split(yuv)
            
            # 应用CLAHE到Y通道
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            y = clahe.apply(y)
            
            # 合并通道
            yuv = cv2.merge([y, u, v])
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            # 灰度图像
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        if not self.parameter_history:
            return {}
        
        # 统计各图像类型的使用频率
        type_counts = {}
        for record in self.parameter_history:
            img_type = record['image_type']
            type_counts[img_type] = type_counts.get(img_type, 0) + 1
        
        return {
            'total_processed': len(self.parameter_history),
            'type_distribution': type_counts,
            'most_common_type': max(type_counts, key=type_counts.get) if type_counts else 'unknown'
        }
    
    def optimize_parameters(self, feedback_data: List[Dict[str, Any]]):
        """
        根据反馈数据优化参数
        
        Args:
            feedback_data: 反馈数据，包含识别结果和准确率
        """
        if not feedback_data:
            return
        
        # 简单的参数优化逻辑
        for feedback in feedback_data:
            image_type = feedback.get('image_type')
            accuracy = feedback.get('accuracy', 0)
            used_params = feedback.get('params', {})
            
            if accuracy < 0.7:  # 准确率低于70%，需要优化
                logger.info(f"优化参数: {image_type}, 准确率: {accuracy:.2f}")
                # 这里可以添加更复杂的优化逻辑
        
        logger.info("参数优化完成")

# 便捷函数
def create_adaptive_threshold_system(config: Optional[Dict[str, Any]] = None) -> AdaptiveThresholdSystem:
    """创建自适应阈值系统实例"""
    return AdaptiveThresholdSystem(config)