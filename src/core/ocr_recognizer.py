"""
OCR识别器模块
集成Tesseract进行文本识别
"""

import logging
import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import pytesseract
from dataclasses import dataclass

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TextRecognitionResult:
    """文本识别结果"""
    text: str
    confidence: float
    position: Tuple[int, int, int, int]  # (x, y, w, h)
    language: str


class OCRRecognizer:
    """OCR识别器类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化OCR识别器
        
        Args:
            config: OCR配置
        """
        self.config = config or {}
        self._init_default_config()
        self._init_tesseract()

        # 检查Tesseract配置
        from src.utils.tesseract_utils import get_tesseract_config
        self.tesseract_config = get_tesseract_config()

        logger.info("OCRRecognizer模块初始化完成")
    
    def _init_default_config(self):
        """初始化默认配置"""
        default_config = {
            'languages': ['eng', 'chi_sim'],  # 支持的语言
            'preprocess': {
                'grayscale': True,
                'denoise': True,
                'threshold': 'adaptive',  # 'adaptive', 'otsu', 'binary', None
                'scale_factor': 2.0,  # 放大因子，提高识别率
                'invert': False,  # 是否反色（白底黑字 -> 黑底白字）
            },
            'ocr_config': {
                'oem': 3,  # 引擎模式: 0-传统, 1-神经网络LSTM, 2-传统+LSTM, 3-默认
                'psm': 6,  # 页面分割模式: 6-假设为统一的文本块
                'dpi': 300,  # 图像DPI
            }
        }
        
        # 合并配置
        for key, value in default_config.items():
            if key not in self.config:
                self.config[key] = value
            elif isinstance(value, dict):
                self.config[key].update({k: v for k, v in value.items() 
                                       if k not in self.config[key]})
    
    def _init_tesseract(self):
        """初始化Tesseract配置"""
        try:
            # 设置Tesseract路径（如果未在smart_automation.py中设置）
            if not hasattr(pytesseract.pytesseract, 'tesseract_cmd'):
                # 从配置文件设置Tesseract
                if self.tesseract_config.get_tesseract_cmd():
                    pytesseract.pytesseract.tesseract_cmd = self.tesseract_config.get_tesseract_cmd()
            
            # 测试Tesseract
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract版本: {version}")
            
            # 获取支持的语言
            langs = pytesseract.get_languages(config='')
            logger.info(f"可用的OCR语言: {langs}")
            
        except Exception as e:
            logger.error(f"Tesseract初始化失败: {e}")
            logger.warning("OCR功能将不可用，请确保Tesseract正确安装")
    
    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        为OCR预处理图像
        
        Args:
            image: 输入图像
            
        Returns:
            预处理后的图像
        """
        preprocess_config = self.config.get('preprocess', {})
        processed = image.copy()
        
        # 1. 转换为灰度图
        if preprocess_config.get('grayscale', True) and len(processed.shape) == 3:
            if processed.shape[2] == 4:  # RGBA
                processed = cv2.cvtColor(processed, cv2.COLOR_RGBA2GRAY)
            else:  # RGB
                processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
        
        # 2. 缩放图像（提高识别率）
        scale_factor = preprocess_config.get('scale_factor', 2.0)
        if scale_factor != 1.0:
            new_width = int(processed.shape[1] * scale_factor)
            new_height = int(processed.shape[0] * scale_factor)
            processed = cv2.resize(processed, (new_width, new_height), 
                                 interpolation=cv2.INTER_CUBIC)
        
        # 3. 去噪
        if preprocess_config.get('denoise', True):
            processed = cv2.medianBlur(processed, 3)
            processed = cv2.GaussianBlur(processed, (3, 3), 0)
        
        # 4. 阈值处理
        threshold_type = preprocess_config.get('threshold', 'adaptive')
        if threshold_type == 'adaptive':
            processed = cv2.adaptiveThreshold(
                processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
        elif threshold_type == 'otsu':
            _, processed = cv2.threshold(processed, 0, 255, 
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif threshold_type == 'binary':
            _, processed = cv2.threshold(processed, 127, 255, cv2.THRESH_BINARY)
        
        # 5. 反色（如果文字是白色背景黑色）
        if preprocess_config.get('invert', False):
            processed = cv2.bitwise_not(processed)
        
        # 6. 形态学操作（可选）
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
        
        return processed
    
    def extract_text(self, 
                    image: np.ndarray, 
                    language: str = 'eng+chi_sim',
                    config: Optional[str] = None) -> TextRecognitionResult:
        """
        从图像中提取文本
        
        Args:
            image: 输入图像
            language: OCR语言
            config: 自定义Tesseract配置
            
        Returns:
            TextRecognitionResult: 识别结果
        """
        try:
            # 预处理图像
            processed_image = self.preprocess_for_ocr(image)
            
            # 获取OCR配置
            ocr_config = self.config.get('ocr_config', {})
            custom_config = f'--oem {ocr_config.get("oem", 3)} --psm {ocr_config.get("psm", 6)} --dpi {ocr_config.get("dpi", 300)}'
            if config:
                custom_config += ' ' + config
            
            # 执行OCR
            data = pytesseract.image_to_data(
                processed_image,
                lang=language,
                config=custom_config,
                output_type=pytesseract.Output.DICT
            )
            
            # 解析结果
            texts = []
            confidences = []
            positions = []
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                if text:  # 只处理非空文本
                    conf = float(data['conf'][i]) if data['conf'][i] != '-1' else 0.0
                    
                    # 只保留置信度高于阈值的文本
                    if conf > 60:  # Tesseract置信度范围0-100
                        texts.append(text)
                        confidences.append(conf)
                        positions.append((
                            data['left'][i],
                            data['top'][i],
                            data['width'][i],
                            data['height'][i]
                        ))
            
            if texts:
                # 合并文本块
                combined_text = ' '.join(texts)
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                
                # 计算整体边界框
                if positions:
                    x_min = min(p[0] for p in positions)
                    y_min = min(p[1] for p in positions)
                    x_max = max(p[0] + p[2] for p in positions)
                    y_max = max(p[1] + p[3] for p in positions)
                    position = (x_min, y_min, x_max - x_min, y_max - y_min)
                else:
                    position = (0, 0, image.shape[1], image.shape[0])
                
                logger.debug(f"OCR识别成功: 文本='{combined_text[:50]}...', "
                           f"置信度={avg_confidence:.1f}")
                
                return TextRecognitionResult(
                    text=combined_text,
                    confidence=avg_confidence,
                    position=position,
                    language=language
                )
            else:
                logger.debug("未识别到文本")
                return TextRecognitionResult(
                    text='',
                    confidence=0.0,
                    position=(0, 0, image.shape[1], image.shape[0]),
                    language=language
                )
                
        except Exception as e:
            logger.error(f"OCR识别失败: {e}")
            return TextRecognitionResult(
                text='',
                confidence=0.0,
                position=(0, 0, image.shape[1], image.shape[0]) if image is not None else (0, 0, 0, 0),
                language=language
            )
    
    def extract_text_from_region(self,
                               image: np.ndarray,
                               region: Tuple[int, int, int, int],
                               language: str = 'eng+chi_sim') -> TextRecognitionResult:
        """
        从图像指定区域提取文本
        
        Args:
            image: 完整图像
            region: 区域 (x, y, width, height)
            language: OCR语言
            
        Returns:
            TextRecognitionResult: 识别结果
        """
        try:
            x, y, w, h = region
            if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
                logger.warning(f"区域超出图像范围: {region}")
                return TextRecognitionResult(
                    text='', confidence=0.0, position=region, language=language
                )
            
            # 提取区域
            roi = image[y:y+h, x:x+w]
            
            # 识别文本
            result = self.extract_text(roi, language)
            
            # 调整位置为原始图像坐标
            adjusted_position = (
                result.position[0] + x,
                result.position[1] + y,
                result.position[2],
                result.position[3]
            )
            
            result.position = adjusted_position
            return result
            
        except Exception as e:
            logger.error(f"区域文本识别失败: {e}")
            return TextRecognitionResult(
                text='', confidence=0.0, position=region, language=language
            )
    
    # def find_text(self,
    #              image: np.ndarray,
    #              target_text: str,
    #              language: str = 'eng+chi_sim',
    #              case_sensitive: bool = False,
    #              similarity_threshold: float = 0.6,
    #              min_confidence: float = 60.0) -> List[TextRecognitionResult]:
    #     """
    #     在图像中查找特定文本
        
    #     Args:
    #         image: 输入图像
    #         target_text: 目标文本
    #         language: OCR语言
    #         case_sensitive: 是否区分大小写
    #         similarity_threshold: 文本相似度阈值
    #         min_confidence: 最小置信度阈值
            
    #     Returns:
    #         找到的文本结果列表
    #     """
    #     # 提取所有文本
    #     result = self.extract_text(image, language)
        
    #     if not result.text:
    #         return []
        
    #     # 简单文本匹配（后续可以改进为更智能的匹配）
    #     found_results = []
        
    #     # 分割文本行（简单实现）
    #     lines = result.text.split('\n') if '\n' in result.text else [result.text]
        
    #     for line in lines:
    #         if not line.strip():
    #             continue
                
    #         # 检查是否包含目标文本
    #         text_to_check = line if case_sensitive else line.lower()
    #         target_to_check = target_text if case_sensitive else target_text.lower()
            
    #         if target_to_check in text_to_check:
    #             # 计算相似度
    #             similarity = self._calculate_text_similarity(
    #                 text_to_check, target_to_check
    #             )
                
    #             if similarity >= similarity_threshold:
    #                 found_results.append(TextRecognitionResult(
    #                     text=line.strip(),
    #                     confidence=result.confidence,
    #                     position=result.position,  # 注意：这里简化了位置
    #                     language=language
    #                 ))
        
    #     return found_results
    
    # def _calculate_text_similarity(self, text1: str, text2: str) -> float:
    #     """计算文本相似度（简单实现）"""
    #     # 简单的字符串包含匹配
    #     if text2 in text1:
    #         return min(1.0, len(text2) / len(text1) + 0.3)
    #     elif text1 in text2:
    #         return min(1.0, len(text1) / len(text2) + 0.3)
    #     else:
    #         # 简单的编辑距离（后续可以改进）
    #         from difflib import SequenceMatcher
    #         return SequenceMatcher(None, text1, text2).ratio()
    def find_text(self,
                image: np.ndarray,
                target_text: str,
                language: str = 'eng+chi_sim',
                case_sensitive: bool = False,
                similarity_threshold: float = 0.6,  # 降低阈值
                min_confidence: float = 60.0) -> List[TextRecognitionResult]:
        """
        在图像中查找特定文本
        
        Args:
            image: 输入图像
            target_text: 目标文本
            language: OCR语言
            case_sensitive: 是否区分大小写
            similarity_threshold: 文本相似度阈值
            min_confidence: 最小置信度阈值
            
        Returns:
            找到的文本结果列表
        """
        try:
            # 预处理图像
            processed_image = self.preprocess_for_ocr(image)
            
            # 获取OCR配置
            ocr_config = self.config.get('ocr_config', {})
            custom_config = f'--oem {ocr_config.get("oem", 3)} --psm {ocr_config.get("psm", 6)}'
            
            # 获取详细的OCR数据
            data = pytesseract.image_to_data(
                processed_image,
                lang=language,
                config=custom_config,
                output_type=pytesseract.Output.DICT
            )
            
            found_results = []
            target_to_check = target_text if case_sensitive else target_text.lower()
            
            # 遍历所有检测到的文本块
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                conf = float(data['conf'][i]) if data['conf'][i] != '-1' else 0.0
                
                if text and conf >= min_confidence:
                    text_to_check = text if case_sensitive else text.lower()
                    
                    # 使用更灵活的匹配方式
                    similarity = self._calculate_text_similarity(text_to_check, target_to_check)
                    
                    if similarity >= similarity_threshold:
                        position = (
                            data['left'][i],
                            data['top'][i],
                            data['width'][i],
                            data['height'][i]
                        )
                        
                        found_results.append(TextRecognitionResult(
                            text=text,
                            confidence=conf,
                            position=position,
                            language=language
                        ))
                        
                        logger.debug(f"找到匹配文本: '{text}' -> '{target_text}' "
                                f"(相似度: {similarity:.2f}, 置信度: {conf:.1f}%)")
            
            if found_results:
                logger.info(f"找到 {len(found_results)} 个匹配文本: '{target_text}'")
            else:
                logger.debug(f"未找到匹配文本: '{target_text}'")
            
            return found_results
            
        except Exception as e:
            logger.error(f"查找文本失败: {e}")
            return []

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        from difflib import SequenceMatcher
        
        # 移除空格和特殊字符
        text1_clean = ''.join(c for c in text1 if c.isalnum())
        text2_clean = ''.join(c for c in text2 if c.isalnum())
        
        if not text1_clean or not text2_clean:
            return 0.0
        
        # 检查是否包含
        if text2_clean in text1_clean:
            return min(1.0, len(text2_clean) / len(text1_clean) + 0.5)
        elif text1_clean in text2_clean:
            return min(1.0, len(text1_clean) / len(text2_clean) + 0.5)
        
        # 使用编辑距离
        return SequenceMatcher(None, text1_clean, text2_clean).ratio()
    
    def save_debug_images(self, 
                         original: np.ndarray, 
                         processed: np.ndarray,
                         prefix: str = "ocr_debug"):
        """
        保存调试图像
        
        Args:
            original: 原始图像
            processed: 处理后的图像
            prefix: 文件名前缀
        """
        import os
        from datetime import datetime
        
        debug_dir = "outputs/ocr_debug"
        os.makedirs(debug_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        original_path = os.path.join(debug_dir, f"{prefix}_original_{timestamp}.png")
        processed_path = os.path.join(debug_dir, f"{prefix}_processed_{timestamp}.png")
        
        cv2.imwrite(original_path, original)
        cv2.imwrite(processed_path, processed)
        
        logger.debug(f"调试图像已保存: {original_path}, {processed_path}")


# 便捷函数
def create_ocr_recognizer(config: Optional[Dict[str, Any]] = None) -> OCRRecognizer:
    """创建OCR识别器实例"""
    return OCRRecognizer(config)