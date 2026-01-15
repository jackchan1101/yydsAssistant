"""
智能自动化核心模块
集成OpenCV和Tesseract，提供高级的图像识别和自动化功能。
"""

import time
import logging
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass

import cv2
import pyautogui
from PIL import ImageGrab, Image
import pytesseract

from ..utils.logger import get_logger

# 设置Tesseract命令路径 (根据你的安装位置调整)
# 注意：这通常在系统环境变量中配置更好，这里作为备用。
# pytesseract.pytesseract.tesseract_cmd = r'D:\DevTools\Tesseract-OCR\tesseract.exe'

# 初始化日志
logger = get_logger(__name__)


@dataclass
class MatchResult:
    """图像匹配结果"""
    found: bool
    position: Optional[Tuple[int, int]] = None  # (left, top) 匹配区域的左上角坐标
    confidence: Optional[float] = None  # 匹配置信度 (0-1)
    method: Optional[str] = None  # 使用的匹配方法
    screen_region: Optional[Tuple[int, int, int, int]] = None  # 搜索区域 (left, top, width, height)
    scale: Optional[float] = None  # 匹配时的缩放比例（仅多尺度匹配）


class SmartAutomation:
    """智能自动化类，集成OpenCV进行图像识别"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化智能自动化模块

        Args:
            config: 配置字典，可包含各种预处理参数
        """
        self.config = config or {}
        self._init_default_config()
        logger.info("SmartAutomation模块初始化完成")

    def _init_default_config(self):
        """初始化默认配置"""
        default_config = {
            'preprocess': {
                'grayscale': True,  # 是否转换为灰度图
                'blur_kernel': (5, 5),  # 高斯模糊核大小
                'threshold_type': 'adaptive',  # 阈值类型: 'adaptive', 'otsu', 'binary'
                'threshold_value': 127,  # 二进制阈值
                'canny_low': 50,  # Canny边缘检测低阈值
                'canny_high': 150,  # Canny边缘检测高阈值
            },
            'matching': {
                'method': cv2.TM_CCOEFF_NORMED,  # 默认匹配方法
                'threshold': 0.8,  # 匹配阈值
                'multi_scale': False,  # 是否启用多尺度匹配
                'scale_range': (0.5, 1.5),  # 尺度搜索范围
                'scale_steps': 5,  # 尺度搜索步数
            }
        }

        # 合并配置
        for key, value in default_config.items():
            if key not in self.config:
                self.config[key] = value
            elif isinstance(value, dict):
                self.config[key].update({k: v for k, v in value.items() if k not in self.config[key]})

    def capture_screen(self, region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        捕获屏幕截图，返回OpenCV格式(numpy数组)的图像

        Args:
            region: 截图区域 (left, top, width, height)，为None时截全屏

        Returns:
            numpy.ndarray: OpenCV格式的BGR图像
        """
        try:
            if region:
                # PIL 的 region 是 (left, top, right, bottom)
                pil_region = (region[0], region[1], region[0] + region[2], region[1] + region[3])
                screenshot = ImageGrab.grab(bbox=pil_region)
            else:
                screenshot = ImageGrab.grab()

            # 转换为OpenCV格式 (BGR)
            open_cv_image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            logger.debug(f"屏幕截图已捕获，尺寸: {open_cv_image.shape}")
            return open_cv_image

        except Exception as e:
            logger.error(f"捕获屏幕截图失败: {e}")
            raise

    def preprocess_image(self, image: np.ndarray, 
                        preprocess_config: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        对图像进行预处理

        Args:
            image: 输入图像 (BGR格式)
            preprocess_config: 预处理配置，为None时使用默认配置

        Returns:
            np.ndarray: 预处理后的图像
        """
        if preprocess_config is None:
            preprocess_config = self.config.get('preprocess', {})

        processed = image.copy()

        # 1. 转换为灰度图
        if preprocess_config.get('grayscale', True):
            if len(processed.shape) == 3:
                processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

        # 2. 高斯模糊 (去噪)
        blur_kernel = preprocess_config.get('blur_kernel')
        if blur_kernel and blur_kernel[0] > 0 and blur_kernel[1] > 0:
            processed = cv2.GaussianBlur(processed, blur_kernel, 0)

        # 3. 阈值处理
        threshold_type = preprocess_config.get('threshold_type', 'adaptive')
        if threshold_type == 'adaptive' and len(processed.shape) == 2:
            processed = cv2.adaptiveThreshold(
                processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        elif threshold_type == 'otsu' and len(processed.shape) == 2:
            _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif threshold_type == 'binary' and len(processed.shape) == 2:
            threshold_value = preprocess_config.get('threshold_value', 127)
            _, processed = cv2.threshold(processed, threshold_value, 255, cv2.THRESH_BINARY)

        # 4. 边缘检测 (可选)
        if preprocess_config.get('canny', False) and len(processed.shape) == 2:
            low = preprocess_config.get('canny_low', 50)
            high = preprocess_config.get('canny_high', 150)
            processed = cv2.Canny(processed, low, high)

        return processed

    def find_image(self, 
                  template_path: str, 
                  screen_region: Optional[Tuple[int, int, int, int]] = None,
                  threshold: Optional[float] = None,
                  preprocess_both: bool = True) -> MatchResult:
        """
        在屏幕上查找模板图像 (使用OpenCV的模板匹配)

        Args:
            template_path: 模板图像路径
            screen_region: 屏幕搜索区域 (left, top, width, height)
            threshold: 匹配阈值，0-1之间，None时使用配置中的阈值
            preprocess_both: 是否对屏幕和模板都进行预处理

        Returns:
            MatchResult: 匹配结果对象
        """
        try:
            # 1. 加载模板图像
            template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
            if template is None:
                logger.error(f"无法加载模板图像: {template_path}")
                return MatchResult(found=False)

            # 2. 捕获屏幕
            screen = self.capture_screen(screen_region)

            # 3. 预处理
            if preprocess_both:
                screen_processed = self.preprocess_image(screen)
                template_processed = self.preprocess_image(template)
            else:
                screen_processed = screen
                template_processed = template

            # 确保图像维度一致
            if len(screen_processed.shape) != len(template_processed.shape):
                if len(screen_processed.shape) == 3:  # 屏幕是彩色，模板是灰度
                    screen_processed = cv2.cvtColor(screen_processed, cv2.COLOR_BGR2GRAY)
                elif len(template_processed.shape) == 3:  # 模板是彩色，屏幕是灰度
                    template_processed = cv2.cvtColor(template_processed, cv2.COLOR_BGR2GRAY)

            # 4. 模板匹配
            match_config = self.config.get('matching', {})
            method = match_config.get('method', cv2.TM_CCOEFF_NORMED)
            match_threshold = threshold or match_config.get('threshold', 0.8)

            result = cv2.matchTemplate(screen_processed, template_processed, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            # 根据匹配方法判断最佳匹配
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                match_val = 1 - min_val
                match_loc = min_loc
            else:
                match_val = max_val
                match_loc = max_loc

            logger.debug(f"匹配完成，最佳匹配值: {match_val:.3f}，阈值: {match_threshold}")

            # 5. 判断是否匹配成功
            if match_val >= match_threshold:
                # 计算匹配位置 (相对于屏幕区域)
                left, top = match_loc
                if screen_region:
                    left += screen_region[0]
                    top += screen_region[1]

                # 获取模板尺寸
                h, w = template.shape[:2]
                center_x = left + w // 2
                center_y = top + h // 2

                logger.info(f"找到匹配图像: {template_path}，位置: ({left}, {top})，置信度: {match_val:.3f}")
                return MatchResult(
                    found=True,
                    position=(center_x, center_y),  # 返回中心点坐标方便点击
                    confidence=float(match_val),
                    method=cv2_tm_method_to_str(method),
                    screen_region=screen_region
                )
            else:
                logger.debug(f"未找到匹配图像: {template_path}，最佳匹配值: {match_val:.3f}，低于阈值: {match_threshold}")
                return MatchResult(found=False, confidence=float(match_val))

        except Exception as e:
            logger.error(f"图像匹配过程中发生错误: {e}")
            return MatchResult(found=False)

    def click_image(self, 
                   template_path: str, 
                   screen_region: Optional[Tuple[int, int, int, int]] = None,
                   threshold: Optional[float] = None,
                   offset: Tuple[int, int] = (0, 0),
                   button: str = 'left',
                   clicks: int = 1,
                   interval: float = 0.1) -> bool:
        """
        查找图像并点击

        Args:
            template_path: 模板图像路径
            screen_region: 屏幕搜索区域
            threshold: 匹配阈值
            offset: 相对于匹配中心的点击偏移
            button: 鼠标按钮 ('left', 'right', 'middle')
            clicks: 点击次数
            interval: 点击间隔

        Returns:
            bool: 是否成功找到并点击
        """
        result = self.find_image(template_path, screen_region, threshold)
        
        if result.found and result.position:
            x, y = result.position
            x += offset[0]
            y += offset[1]
            
            pyautogui.click(x=x, y=y, button=button, clicks=clicks, interval=interval)
            logger.info(f"已点击图像位置: ({x}, {y})")
            return True
        else:
            logger.warning(f"未找到图像，无法点击: {template_path}")
            return False

    def find_all_images(self, 
                       template_path: str, 
                       screen_region: Optional[Tuple[int, int, int, int]] = None,
                       threshold: Optional[float] = None) -> list:
        """
        查找屏幕上所有匹配的模板图像

        Args:
            template_path: 模板图像路径
            screen_region: 屏幕搜索区域
            threshold: 匹配阈值

        Returns:
            list: 所有匹配结果的列表，每个元素为MatchResult
        """
        try:
            # 加载模板
            template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
            if template is None:
                logger.error(f"无法加载模板图像: {template_path}")
                return []

            # 捕获屏幕
            screen = self.capture_screen(screen_region)
            screen_processed = self.preprocess_image(screen)
            template_processed = self.preprocess_image(template)

            # 确保图像维度一致
            if len(screen_processed.shape) != len(template_processed.shape):
                if len(screen_processed.shape) == 3:
                    screen_processed = cv2.cvtColor(screen_processed, cv2.COLOR_BGR2GRAY)
                elif len(template_processed.shape) == 3:
                    template_processed = cv2.cvtColor(template_processed, cv2.COLOR_BGR2GRAY)

            # 模板匹配
            match_config = self.config.get('matching', {})
            method = match_config.get('method', cv2.TM_CCOEFF_NORMED)
            match_threshold = threshold or match_config.get('threshold', 0.8)

            result = cv2.matchTemplate(screen_processed, template_processed, method)
            
            # 获取所有超过阈值的匹配位置
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                locations = np.where(result <= 1 - match_threshold)
            else:
                locations = np.where(result >= match_threshold)

            matches = []
            h, w = template.shape[:2]
            
            # 使用非极大值抑制去除重叠的匹配
            for pt in zip(*locations[::-1]):
                # 检查与已有匹配是否重叠
                overlap = False
                for match in matches:
                    existing_pt = match.position
                    if existing_pt:
                        # 计算两个匹配之间的距离
                        distance = np.sqrt((pt[0] - existing_pt[0])**2 + (pt[1] - existing_pt[1])**2)
                        if distance < max(w, h) * 0.5:  # 如果距离小于模板尺寸的一半，认为是重叠
                            overlap = True
                            break
                
                if not overlap:
                    # 计算置信度
                    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                        confidence = 1 - result[pt[1], pt[0]]
                    else:
                        confidence = result[pt[1], pt[0]]
                    
                    # 计算中心点
                    center_x = pt[0] + w // 2
                    center_y = pt[1] + h // 2
                    
                    if screen_region:
                        center_x += screen_region[0]
                        center_y += screen_region[1]
                    
                    matches.append(MatchResult(
                        found=True,
                        position=(center_x, center_y),
                        confidence=float(confidence),
                        method=cv2_tm_method_to_str(method),
                        screen_region=screen_region
                    ))

            logger.info(f"找到 {len(matches)} 个匹配")
            return matches

        except Exception as e:
            logger.error(f"查找所有图像时发生错误: {e}")
            return []
    # 在多尺度下搜索模板图像
    def find_image_multi_scale(self, 
                            template_path: str, 
                            screen_region: Optional[Tuple[int, int, int, int]] = None,
                            threshold: Optional[float] = None,
                            scale_range: Tuple[float, float] = (0.5, 2.0),
                            scale_steps: int = 20) -> MatchResult:
        """
        在多尺度下搜索模板图像
        
        Args:
            template_path: 模板图像路径
            screen_region: 屏幕搜索区域
            threshold: 匹配阈值
            scale_range: 缩放范围 (最小比例, 最大比例)
            scale_steps: 缩放步数
            
        Returns:
            MatchResult: 最佳匹配结果
        """
        try:
            # 1. 加载模板
            template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
            if template is None:
                logger.error(f"无法加载模板图像: {template_path}")
                return MatchResult(found=False)
            
            # 2. 捕获屏幕
            screen = self.capture_screen(screen_region)
            
            # 3. 预处理
            screen_processed = self.preprocess_image(screen)
            template_processed = self.preprocess_image(template)
            
            # 确保图像维度一致
            if len(screen_processed.shape) != len(template_processed.shape):
                if len(screen_processed.shape) == 3:
                    screen_processed = cv2.cvtColor(screen_processed, cv2.COLOR_BGR2GRAY)
                elif len(template_processed.shape) == 3:
                    template_processed = cv2.cvtColor(template_processed, cv2.COLOR_BGR2GRAY)
            
            # 4. 获取匹配配置
            match_config = self.config.get('matching', {})
            method = match_config.get('method', cv2.TM_CCOEFF_NORMED)
            match_threshold = threshold or match_config.get('threshold', 0.8)
            
            # 5. 多尺度搜索
            best_match = MatchResult(found=False, confidence=0.0)
            original_h, original_w = template_processed.shape[:2]
            
            for scale in np.linspace(scale_range[0], scale_range[1], scale_steps):
                # 计算当前尺度的模板尺寸
                width = int(original_w * scale)
                height = int(original_h * scale)
                
                # 跳过无效尺寸
                if width < 10 or height < 10 or width > screen_processed.shape[1] or height > screen_processed.shape[0]:
                    continue
                
                # 缩放模板
                scaled_template = cv2.resize(template_processed, (width, height), 
                                            interpolation=cv2.INTER_AREA)
                
                # 模板匹配
                result = cv2.matchTemplate(screen_processed, scaled_template, method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                # 根据匹配方法判断最佳匹配
                if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                    match_val = 1 - min_val
                    match_loc = min_loc
                else:
                    match_val = max_val
                    match_loc = max_loc
                
                # 更新最佳匹配
                if match_val > best_match.confidence:
                    if match_val >= match_threshold:
                        # 计算实际位置
                        left, top = match_loc
                        center_x = left + width // 2
                        center_y = top + height // 2
                        
                        if screen_region:
                            center_x += screen_region[0]
                            center_y += screen_region[1]
                        
                        best_match = MatchResult(
                            found=True,
                            position=(center_x, center_y),
                            confidence=float(match_val),
                            method=f"MultiScale-{cv2_tm_method_to_str(method)}",
                            screen_region=screen_region,
                            scale=scale
                        )
                    else:
                        best_match.confidence = float(match_val)
            
            if best_match.found:
                logger.info(f"多尺度匹配成功: {template_path}, 尺度: {best_match.scale:.2f}, "
                        f"置信度: {best_match.confidence:.3f}")
            else:
                logger.debug(f"多尺度匹配失败: {template_path}, 最佳匹配值: {best_match.confidence:.3f}")
            
            return best_match
            
        except Exception as e:
            logger.error(f"多尺度匹配过程中发生错误: {e}")
            return MatchResult(found=False)
    # 特征点匹配
    def find_image_with_features(self, 
                            template_path: str, 
                            screen_region: Optional[Tuple[int, int, int, int]] = None,
                            method: str = 'orb',
                            min_matches: int = 10,
                            ratio_test: float = 0.75) -> MatchResult:
        """
        使用特征点匹配算法（支持旋转和尺度不变）
        
        Args:
            template_path: 模板图像路径
            screen_region: 屏幕搜索区域
            method: 特征检测方法 ('orb', 'sift', 'brisk')
            min_matches: 最小匹配点数
            ratio_test: 比率测试阈值
        
        Returns:
            MatchResult: 匹配结果
        """
        try:
            # 1. 加载模板
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                logger.error(f"无法加载模板图像: {template_path}")
                return MatchResult(found=False)
            
            # 2. 捕获屏幕
            screen = self.capture_screen(screen_region)
            screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY) if len(screen.shape) == 3 else screen
            
            # 3. 初始化特征检测器
            if method.lower() == 'orb':
                detector = cv2.ORB_create(nfeatures=1000)
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            elif method.lower() == 'sift' and hasattr(cv2, 'SIFT_create'):
                detector = cv2.SIFT_create()
                matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            elif method.lower() == 'brisk':
                detector = cv2.BRISK_create()
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            else:
                logger.error(f"不支持的特征检测方法: {method}")
                return MatchResult(found=False)
            
            # 4. 检测特征点和计算描述符
            kp1, des1 = detector.detectAndCompute(template, None)
            kp2, des2 = detector.detectAndCompute(screen_gray, None)
            
            if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
                logger.debug(f"特征点不足: 模板{len(kp1) if kp1 else 0}个, 屏幕{len(kp2) if kp2 else 0}个")
                return MatchResult(found=False, confidence=0.0, method=f"Feature-{method}")
            
            # 5. 特征匹配
            matches = matcher.knnMatch(des1, des2, k=2)
            
            # 6. 应用比率测试
            good_matches = []
            for m, n in matches:
                if m.distance < ratio_test * n.distance:
                    good_matches.append(m)
            
            # 7. 判断是否匹配成功
            match_ratio = len(good_matches) / len(matches) if matches else 0
            confidence = min(len(good_matches) / min_matches, 1.0)  # 归一化置信度
            
            logger.debug(f"特征匹配: 总匹配数={len(matches)}, 好匹配数={len(good_matches)}, "
                        f"匹配比例={match_ratio:.3f}, 置信度={confidence:.3f}")
            
            if len(good_matches) >= min_matches:
                # 8. 计算单应性矩阵
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if H is not None:
                    # 9. 计算模板在屏幕中的位置
                    h, w = template.shape
                    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, H)
                    
                    # 计算中心点
                    center_x = int(np.mean(dst[:, 0, 0]))
                    center_y = int(np.mean(dst[:, 0, 1]))
                    
                    if screen_region:
                        center_x += screen_region[0]
                        center_y += screen_region[1]
                    
                    logger.info(f"特征匹配成功: {template_path}, 方法={method}, "
                            f"匹配点数={len(good_matches)}, 置信度={confidence:.3f}")
                    
                    return MatchResult(
                        found=True,
                        position=(center_x, center_y),
                        confidence=float(confidence),
                        method=f"Feature-{method}",
                        screen_region=screen_region,
                        homography=H,
                        matches_count=len(good_matches)
                    )
            
            return MatchResult(
                found=False,
                confidence=float(confidence),
                method=f"Feature-{method}",
                matches_count=len(good_matches)
            )
            
        except Exception as e:
            logger.error(f"特征匹配过程中发生错误: {e}")
            return MatchResult(found=False)
    # 智能匹配器
    def smart_find_image(self, 
                        template_path: str, 
                        screen_region: Optional[Tuple[int, int, int, int]] = None,
                        methods: List[str] = None) -> MatchResult:
        """
        智能图像匹配：自动选择最佳匹配算法
        
        算法选择策略：
        1. 先尝试标准模板匹配（最快）
        2. 如果失败，尝试多尺度模板匹配
        3. 如果还失败，尝试特征匹配
        
        Args:
            template_path: 模板图像路径
            screen_region: 屏幕搜索区域
            methods: 要尝试的方法列表，None时使用默认策略
            
        Returns:
            MatchResult: 最佳匹配结果
        """
        if methods is None:
            methods = ['template', 'multi_scale', 'orb']
        
        best_result = MatchResult(found=False, confidence=0.0)
        
        for method in methods:
            try:
                if method == 'template':
                    result = self.find_image(template_path, screen_region)
                elif method == 'multi_scale':
                    result = self.find_image_multi_scale(template_path, screen_region)
                elif method in ['orb', 'sift', 'brisk']:
                    result = self.find_image_with_features(template_path, screen_region, method)
                else:
                    logger.warning(f"未知的匹配方法: {method}")
                    continue
                
                # 更新最佳结果
                if result.found and result.confidence > best_result.confidence:
                    best_result = result
                
                # 如果找到高置信度匹配，提前返回
                if result.found and result.confidence >= 0.9:
                    logger.info(f"智能匹配: 使用{method}找到高置信度匹配")
                    return result
                    
            except Exception as e:
                logger.warning(f"方法 {method} 失败: {e}")
                continue
        
        logger.info(f"智能匹配完成: 最佳方法={best_result.method}, "
                f"置信度={best_result.confidence:.3f}")
        return best_result

def cv2_tm_method_to_str(method: int) -> str:
    """将OpenCV模板匹配方法转换为字符串表示"""
    methods = {
        cv2.TM_CCOEFF: 'TM_CCOEFF',
        cv2.TM_CCOEFF_NORMED: 'TM_CCOEFF_NORMED',
        cv2.TM_CCORR: 'TM_CCORR',
        cv2.TM_CCORR_NORMED: 'TM_CCORR_NORMED',
        cv2.TM_SQDIFF: 'TM_SQDIFF',
        cv2.TM_SQDIFF_NORMED: 'TM_SQDIFF_NORMED'
    }
    return methods.get(method, f'UNKNOWN({method})')