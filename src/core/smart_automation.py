"""
智能自动化核心模块，集成OpenCV和Tesseract的高级自动化功能
"""
import pyautogui
import cv2
import numpy as np
from PIL import ImageGrab
import time
import logging
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import asdict

from src.utils.opencv_utils import OpenCVProcessor
from src.utils.config_manager import ConfigManager, MatchConfig, ClickConfig

class SmartAutomation:
    """智能自动化类"""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        初始化智能自动化
        
        Args:
            config_manager: 配置管理器实例
        """
        self.config_manager = config_manager or ConfigManager()
        self.cv_processor = OpenCVProcessor(
            debug_mode=self.config_manager.get("general.debug_mode", False)
        )
        self.logger = logging.getLogger(__name__)
        
        # 性能监控
        self.recognition_times = []
        
    def capture_screen(self, region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        捕获屏幕
        
        Args:
            region: 捕获区域 (left, top, width, height)
            
        Returns:
            屏幕图像
        """
        screenshot = pyautogui.screenshot(region=region)
        return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    def find_image_position(self, 
                          template_path: str,
                          region: Optional[Tuple[int, int, int, int]] = None,
                          match_config: Optional[MatchConfig] = None) -> Optional[Tuple[int, int]]:
        """
        查找图像在屏幕中的位置
        
        Args:
            template_path: 模板图像路径
            region: 搜索区域
            match_config: 匹配配置
            
        Returns:
            图像中心位置或None
        """
        start_time = time.time()
        
        try:
            # 加载模板图像
            template_img = cv2.imread(template_path)
            if template_img is None:
                self.logger.error(f"无法加载模板图像: {template_path}")
                return None
                
            # 捕获屏幕
            screen_img = self.capture_screen(region)
            
            # 应用匹配配置
            if match_config is None:
                match_config = MatchConfig()
                
            # 预处理图像
            preprocessed_screen = self.cv_processor.preprocess_image(
                screen_img, match_config.preprocess_method
            )
            preprocessed_template = self.cv_processor.preprocess_image(
                template_img, match_config.preprocess_method
            )
            
            # 根据方法进行匹配
            if match_config.method.value == "template":
                result = self.cv_processor.match_template(
                    preprocessed_screen, 
                    preprocessed_template,
                    threshold=match_config.threshold
                )
                if result:
                    pos, confidence = result
                    self.logger.debug(f"模板匹配成功: 位置={pos}, 置信度={confidence:.2f}")
                    
                    # 记录性能
                    elapsed = time.time() - start_time
                    self.recognition_times.append(elapsed)
                    
                    if region:
                        # 调整到屏幕绝对坐标
                        pos = (pos[0] + region[0], pos[1] + region[1])
                        
                    return pos
                    
            elif match_config.method.value.startswith("feature"):
                method = match_config.method.value.split("_")[1]
                pos = self.cv_processor.match_features(
                    screen_img, template_img, method
                )
                if pos:
                    self.logger.debug(f"特征匹配成功: 位置={pos}, 方法={method}")
                    
                    # 记录性能
                    elapsed = time.time() - start_time
                    self.recognition_times.append(elapsed)
                    
                    if region:
                        # 调整到屏幕绝对坐标
                        pos = (pos[0] + region[0], pos[1] + region[1])
                        
                    return pos
                    
            elif match_config.method.value == "adaptive":
                result = self.cv_processor.adaptive_match(
                    screen_img, template_img
                )
                if result:
                    pos, method, confidence = result
                    self.logger.debug(f"自适应匹配成功: 位置={pos}, 方法={method}, 置信度={confidence:.2f}")
                    
                    # 记录性能
                    elapsed = time.time() - start_time
                    self.recognition_times.append(elapsed)
                    
                    if region:
                        # 调整到屏幕绝对坐标
                        pos = (pos[0] + region[0], pos[1] + region[1])
                        
                    return pos
                    
        except Exception as e:
            self.logger.error(f"图像查找失败: {e}")
            
        return None
    
    def find_text_position(self, 
                          text: str,
                          region: Optional[Tuple[int, int, int, int]] = None,
                          lang: str = "chi_sim+eng") -> Optional[Tuple[int, int]]:
        """
        查找文本在屏幕中的位置
        
        Args:
            text: 要查找的文本
            region: 搜索区域
            lang: OCR语言
            
        Returns:
            文本中心位置或None
        """
        start_time = time.time()
        
        try:
            # 捕获屏幕
            screen_img = self.capture_screen(region)
            
            # 查找文本
            pos = self.cv_processor.find_text_position(screen_img, text, lang)
            
            if pos:
                self.logger.debug(f"文本查找成功: 文本='{text}', 位置={pos}")
                
                # 记录性能
                elapsed = time.time() - start_time
                self.recognition_times.append(elapsed)
                
                if region:
                    # 调整到屏幕绝对坐标
                    pos = (pos[0] + region[0], pos[1] + region[1])
                    
                return pos
                
        except Exception as e:
            self.logger.error(f"文本查找失败: {e}")
            
        return None
    
    def smart_click(self, click_config: ClickConfig) -> bool:
        """
        智能点击 - 根据配置自动识别并点击
        
        Args:
            click_config: 点击配置
            
        Returns:
            是否成功点击
        """
        try:
            # 等待点击前延迟
            if click_config.delay_before > 0:
                time.sleep(click_config.delay_before)
                
            # 确定点击位置
            if click_config.position:
                # 使用固定位置
                pos = click_config.position
                self.logger.debug(f"使用固定位置: {pos}")
                
            elif click_config.image_path:
                # 通过图像识别确定位置
                pos = self.find_image_position(
                    click_config.image_path,
                    match_config=click_config.match_config
                )
                if not pos:
                    self.logger.warning(f"未找到图像: {click_config.image_path}")
                    return False
                    
            elif click_config.text:
                # 通过文本识别确定位置
                pos = self.find_text_position(
                    click_config.text,
                    lang=click_config.match_config.ocr_lang if click_config.match_config else "chi_sim+eng"
                )
                if not pos:
                    self.logger.warning(f"未找到文本: {click_config.text}")
                    return False
                    
            else:
                self.logger.error("没有指定点击位置、图像或文本")
                return False
                
            # 添加随机偏移
            if click_config.random_offset > 0:
                import random
                offset_x = random.randint(-click_config.random_offset, click_config.random_offset)
                offset_y = random.randint(-click_config.random_offset, click_config.random_offset)
                pos = (pos[0] + offset_x, pos[1] + offset_y)
                
            # 执行点击
            pyautogui.click(pos[0], pos[1])
            self.logger.info(f"点击位置: {pos}")
            
            # 等待点击后延迟
            if click_config.delay_after > 0:
                time.sleep(click_config.delay_after)
                
            return True
            
        except Exception as e:
            self.logger.error(f"智能点击失败: {e}")
            return False
    
    def execute_task(self, task: Dict[str, Any]) -> bool:
        """
        执行任务
        
        Args:
            task: 任务配置
            
        Returns:
            任务是否成功
        """
        task_name = task.get("name", "未命名任务")
        self.logger.info(f"开始执行任务: {task_name}")
        
        try:
            # 解析点击配置
            click_config = ClickConfig(
                position=tuple(task["position"]) if "position" in task else None,
                image_path=task.get("image_path"),
                text=task.get("text"),
                delay_before=task.get("delay_before", 0.5),
                delay_after=task.get("delay_after", 1.0),
                random_offset=task.get("random_offset", 5)
            )
            
            # 解析匹配配置
            if "match_config" in task:
                match_cfg = task["match_config"]
                click_config.match_config = MatchConfig(
                    method=match_cfg.get("method", "adaptive"),
                    threshold=match_cfg.get("threshold", 0.8),
                    preprocess_method=match_cfg.get("preprocess_method", "adaptive"),
                    use_ocr=match_cfg.get("use_ocr", False),
                    ocr_lang=match_cfg.get("ocr_lang", "chi_sim+eng")
                )
                
            # 获取最大尝试次数
            max_attempts = task.get("max_attempts", 
                                  self.config_manager.get("performance.max_attempts", 3))
            
            # 执行点击
            for attempt in range(max_attempts):
                if attempt > 0:
                    retry_delay = task.get("retry_delay", 
                                         self.config_manager.get("performance.retry_delay", 1.0))
                    self.logger.debug(f"第 {attempt} 次重试，等待 {retry_delay} 秒")
                    time.sleep(retry_delay)
                    
                if self.smart_click(click_config):
                    self.logger.info(f"任务执行成功: {task_name}")
                    return True
                    
            self.logger.warning(f"任务执行失败: {task_name}，已达到最大尝试次数 {max_attempts}")
            return False
            
        except Exception as e:
            self.logger.error(f"任务执行异常: {task_name}, 错误: {e}")
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取性能统计
        
        Returns:
            性能统计数据
        """
        if not self.recognition_times:
            return {"avg_recognition_time": 0, "total_recognitions": 0}
            
        avg_time = sum(self.recognition_times) / len(self.recognition_times)
        return {
            "avg_recognition_time": avg_time,
            "total_recognitions": len(self.recognition_times),
            "recognition_times": self.recognition_times[:10]  # 最近10次
        }