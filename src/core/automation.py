"""
自动化核心模块，提供屏幕识别和点击等基础自动化功能
"""
import time
import pyautogui
from typing import Tuple, Optional, Dict, Any
from PIL import Image
import os
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.config_manager import get_config_manager

logger = get_logger(__name__)


class AutomationCore:
    """自动化核心类"""
    
    def __init__(self, config_dir: str = "./configs"):
        """
        初始化自动化核心
        
        Args:
            config_dir: 配置目录
        """
        self.config_manager = get_config_manager(config_dir)
        self.settings = self.config_manager.load_config("settings.yaml")
        
        # 初始化pyautogui设置
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1  # 每次pyautogui调用后的暂停时间
        
        # 获取配置参数
        self.confidence = self.settings.get("automation", {}).get("confidence", 0.8)
        self.click_delay = self.settings.get("automation", {}).get("click_delay", 0.5)
        self.retry_times = self.settings.get("automation", {}).get("retry_times", 3)
        self.retry_interval = self.settings.get("automation", {}).get("retry_interval", 1.0)
        
        logger.info("自动化核心初始化完成")
        logger.debug(f"配置参数: confidence={self.confidence}, click_delay={self.click_delay}")
    
    def screenshot(self, 
                   region: Optional[Tuple[int, int, int, int]] = None,
                   save_path: Optional[str] = None) -> Image.Image:
        """
        截取屏幕截图
        
        Args:
            region: 截图区域 (left, top, width, height)，为None时截全屏
            save_path: 截图保存路径，为None时不保存
            
        Returns:
            PIL Image对象
        """
        try:
            screenshot = pyautogui.screenshot(region=region)
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                screenshot.save(save_path)
                logger.debug(f"截图已保存: {save_path}")
            
            return screenshot
            
        except Exception as e:
            logger.error(f"截图失败: {e}")
            raise
    
    def find_image(self, 
                   template_path: str,
                   region: Optional[Tuple[int, int, int, int]] = None,
                   confidence: Optional[float] = None,
                   grayscale: bool = True) -> Optional[Tuple[int, int, int, int]]:
        """
        在屏幕上查找图片
        
        Args:
            template_path: 模板图片路径
            region: 搜索区域，None表示全屏搜索
            confidence: 匹配置信度，None使用默认值
            grayscale: 是否使用灰度匹配
            
        Returns:
            找到的位置 (left, top, width, height)，未找到返回None
        """
        if confidence is None:
            confidence = self.confidence
        
        try:
            # 检查模板图片是否存在
            if not os.path.exists(template_path):
                logger.error(f"模板图片不存在: {template_path}")
                return None
            
            # 在屏幕上查找图片
            location = pyautogui.locateOnScreen(
                template_path,
                region=region,
                confidence=confidence,
                grayscale=grayscale
            )
            
            if location:
                logger.debug(f"找到图片: {template_path} 位置: {location}")
                return location
            else:
                logger.debug(f"未找到图片: {template_path}")
                return None
                
        except Exception as e:
            logger.error(f"查找图片失败 {template_path}: {e}")
            return None
    
    def find_image_center(self, 
                          template_path: str,
                          region: Optional[Tuple[int, int, int, int]] = None,
                          confidence: Optional[float] = None) -> Optional[Tuple[int, int]]:
        """
        查找图片并返回中心点坐标
        
        Args:
            template_path: 模板图片路径
            region: 搜索区域
            confidence: 匹配置信度
            
        Returns:
            图片中心点坐标 (x, y)，未找到返回None
        """
        location = self.find_image(template_path, region, confidence)
        
        if location:
            center_x = location.left + location.width // 2
            center_y = location.top + location.height // 2
            return center_x, center_y
        
        return None
    
    def click(self, 
              x: int, 
              y: int, 
              button: str = 'left',
              clicks: int = 1,
              interval: Optional[float] = None,
              delay_before: Optional[float] = None,
              delay_after: Optional[float] = None) -> bool:
        """
        在指定位置点击
        
        Args:
            x: X坐标
            y: Y坐标
            button: 鼠标按钮 ('left', 'middle', 'right')
            clicks: 点击次数
            interval: 多次点击之间的间隔
            delay_before: 点击前的延迟
            delay_after: 点击后的延迟
            
        Returns:
            是否点击成功
        """
        try:
            if delay_before:
                time.sleep(delay_before)
            
            if interval is None:
                interval = self.click_delay / 2
            
            pyautogui.click(x=x, y=y, button=button, clicks=clicks, interval=interval)
            
            if delay_after:
                time.sleep(delay_after)
            
            logger.debug(f"点击位置: ({x}, {y}), 按钮: {button}, 次数: {clicks}")
            return True
            
        except Exception as e:
            logger.error(f"点击失败 ({x}, {y}): {e}")
            return False
    
    def click_image(self, 
                    template_path: str,
                    button: str = 'left',
                    clicks: int = 1,
                    region: Optional[Tuple[int, int, int, int]] = None,
                    confidence: Optional[float] = None,
                    retry: bool = True) -> bool:
        """
        查找并点击图片
        
        Args:
            template_path: 模板图片路径
            button: 鼠标按钮
            clicks: 点击次数
            region: 搜索区域
            confidence: 匹配置信度
            retry: 是否重试
            
        Returns:
            是否点击成功
        """
        max_retries = self.retry_times if retry else 1
        
        for attempt in range(max_retries):
            center = self.find_image_center(template_path, region, confidence)
            
            if center:
                x, y = center
                if self.click(x, y, button, clicks):
                    logger.info(f"成功点击图片: {template_path}")
                    return True
                else:
                    logger.warning(f"找到图片但点击失败: {template_path}")
            else:
                logger.debug(f"未找到图片，重试 {attempt + 1}/{max_retries}: {template_path}")
            
            if attempt < max_retries - 1:
                time.sleep(self.retry_interval)
        
        logger.warning(f"点击图片失败: {template_path}")
        return False
    
    def move_to(self, 
                x: int, 
                y: int, 
                duration: Optional[float] = None,
                delay_before: Optional[float] = None,
                delay_after: Optional[float] = None) -> bool:
        """
        移动鼠标到指定位置
        
        Args:
            x: X坐标
            y: Y坐标
            duration: 移动持续时间
            delay_before: 移动前的延迟
            delay_after: 移动后的延迟
            
        Returns:
            是否移动成功
        """
        try:
            if delay_before:
                time.sleep(delay_before)
            
            pyautogui.moveTo(x, y, duration=duration)
            
            if delay_after:
                time.sleep(delay_after)
            
            logger.debug(f"鼠标移动到: ({x}, {y})")
            return True
            
        except Exception as e:
            logger.error(f"移动鼠标失败 ({x}, {y}): {e}")
            return False
    
    def move_to_image(self, 
                      template_path: str,
                      region: Optional[Tuple[int, int, int, int]] = None,
                      confidence: Optional[float] = None,
                      duration: Optional[float] = None) -> bool:
        """
        移动鼠标到图片位置
        
        Args:
            template_path: 模板图片路径
            region: 搜索区域
            confidence: 匹配置信度
            duration: 移动持续时间
            
        Returns:
            是否移动成功
        """
        center = self.find_image_center(template_path, region, confidence)
        
        if center:
            x, y = center
            return self.move_to(x, y, duration)
        
        logger.warning(f"移动鼠标到图片失败，未找到图片: {template_path}")
        return False
    
    def get_mouse_position(self) -> Tuple[int, int]:
        """
        获取当前鼠标位置
        
        Returns:
            鼠标坐标 (x, y)
        """
        return pyautogui.position()
    
    def wait_until_image_appears(self, 
                                 template_path: str,
                                 timeout: float = 10.0,
                                 region: Optional[Tuple[int, int, int, int]] = None,
                                 confidence: Optional[float] = None,
                                 check_interval: float = 0.5) -> Optional[Tuple[int, int]]:
        """
        等待直到图片出现
        
        Args:
            template_path: 模板图片路径
            timeout: 超时时间（秒）
            region: 搜索区域
            confidence: 匹配置信度
            check_interval: 检查间隔
            
        Returns:
            找到的图片中心点坐标，超时返回None
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            center = self.find_image_center(template_path, region, confidence)
            
            if center:
                logger.info(f"图片出现: {template_path} 位置: {center}")
                return center
            
            time.sleep(check_interval)
        
        logger.warning(f"等待图片超时: {template_path}")
        return None
    
    def wait_until_image_disappears(self, 
                                    template_path: str,
                                    timeout: float = 10.0,
                                    region: Optional[Tuple[int, int, int, int]] = None,
                                    confidence: Optional[float] = None,
                                    check_interval: float = 0.5) -> bool:
        """
        等待直到图片消失
        
        Args:
            template_path: 模板图片路径
            timeout: 超时时间（秒）
            region: 搜索区域
            confidence: 匹配置信度
            check_interval: 检查间隔
            
        Returns:
            图片是否消失
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            location = self.find_image(template_path, region, confidence)
            
            if not location:
                logger.info(f"图片消失: {template_path}")
                return True
            
            time.sleep(check_interval)
        
        logger.warning(f"等待图片消失超时: {template_path}")
        return False
    
    def drag_to(self, 
                start_x: int, 
                start_y: int, 
                end_x: int, 
                end_y: int,
                duration: float = 1.0,
                button: str = 'left') -> bool:
        """
        从起点拖拽到终点
        
        Args:
            start_x: 起点X坐标
            start_y: 起点Y坐标
            end_x: 终点X坐标
            end_y: 终点Y坐标
            duration: 拖拽持续时间
            button: 鼠标按钮
            
        Returns:
            是否拖拽成功
        """
        try:
            pyautogui.mouseDown(start_x, start_y, button=button)
            time.sleep(0.1)  # 短暂停顿确保按下事件生效
            pyautogui.moveTo(end_x, end_y, duration=duration)
            pyautogui.mouseUp(end_x, end_y, button=button)
            
            logger.debug(f"拖拽从 ({start_x}, {start_y}) 到 ({end_x}, {end_y})")
            return True
            
        except Exception as e:
            logger.error(f"拖拽失败: {e}")
            return False


# 便捷函数
def create_automation(config_dir: str = "./configs") -> AutomationCore:
    """
    创建自动化核心实例
    
    Args:
        config_dir: 配置目录
        
    Returns:
        自动化核心实例
    """
    return AutomationCore(config_dir)


# 测试函数
def test_basic_functions():
    """测试基础功能"""
    logger = get_logger(__name__)
    logger.info("开始测试基础功能...")
    
    automation = create_automation()
    
    # 测试截图
    try:
        screenshot = automation.screenshot()
        logger.info(f"截图成功，大小: {screenshot.size}")
    except Exception as e:
        logger.error(f"截图测试失败: {e}")
    
    # 测试获取鼠标位置
    pos = automation.get_mouse_position()
    logger.info(f"当前鼠标位置: {pos}")
    
    # 测试移动鼠标
    test_x, test_y = pos[0] + 10, pos[1] + 10
    if automation.move_to(test_x, test_y, duration=0.5):
        logger.info(f"鼠标移动测试成功")
    else:
        logger.error("鼠标移动测试失败")
    
    logger.info("基础功能测试完成")

def click(x: int, y: int, button: str = 'left', clicks: int = 1) -> bool:
    """
    在指定位置点击（便捷函数）
    
    Args:
        x: X坐标
        y: Y坐标
        button: 鼠标按钮
        clicks: 点击次数
        
    Returns:
        是否点击成功
    """
    automation = AutomationCore()
    return automation.click(x, y, button=button, clicks=clicks)


def find_image(template_path: str, 
               region: tuple = None, 
               confidence: float = 0.8) -> Optional[Tuple[int, int, int, int]]:
    """
    在屏幕上查找图片（便捷函数）
    
    Args:
        template_path: 模板图片路径
        region: 搜索区域
        confidence: 匹配置信度
        
    Returns:
        找到的位置 (left, top, width, height)
    """
    automation = AutomationCore()
    return automation.find_image(template_path, region=region, confidence=confidence)


def capture_screen(region: tuple = None) -> Image.Image:
    """
    截取屏幕截图（便捷函数）
    
    Args:
        region: 截图区域 (left, top, width, height)
        
    Returns:
        PIL Image对象
    """
    automation = AutomationCore()
    return automation.screenshot(region=region)

if __name__ == "__main__":
    from ..utils.logger import setup_logger
    
    # 设置日志
    setup_logger(enable_file=False, enable_console=True)
    
    # 运行测试
    test_basic_functions()