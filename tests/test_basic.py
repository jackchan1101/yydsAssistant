#!/usr/bin/env python3
"""
基础功能测试
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
from src.utils.logger import setup_logger, get_logger
from src.core.automation import AutomationCore


def test_screenshot():
    """测试截图功能"""
    logger = get_logger(__name__)
    automation = AutomationCore()
    
    logger.info("测试截图功能...")
    
    # 测试全屏截图
    screenshot = automation.screenshot()
    logger.info(f"全屏截图大小: {screenshot.size}")
    
    # 测试区域截图
    screenshot = automation.screenshot(region=(0, 0, 800, 600))
    logger.info(f"区域截图大小: {screenshot.size}")
    
    # 测试保存截图
    test_save_path = project_root / "screenshots" / "test_screenshot.png"
    screenshot = automation.screenshot(save_path=str(test_save_path))
    logger.info(f"截图已保存到: {test_save_path}")
    
    logger.info("截图测试完成")


def test_mouse_operations():
    """测试鼠标操作"""
    logger = get_logger(__name__)
    automation = AutomationCore()
    
    logger.info("测试鼠标操作...")
    
    # 获取当前鼠标位置
    original_pos = automation.get_mouse_position()
    logger.info(f"原始鼠标位置: {original_pos}")
    
    # 测试移动鼠标
    target_pos = (original_pos[0] + 100, original_pos[1] + 100)
    if automation.move_to(target_pos[0], target_pos[1], duration=1.0):
        logger.info(f"鼠标移动到: {target_pos}")
    
    # 等待一会儿
    time.sleep(1)
    
    # 测试点击
    if automation.click(target_pos[0], target_pos[1], delay_after=0.5):
        logger.info("点击测试成功")
    
    # 返回原始位置
    automation.move_to(original_pos[0], original_pos[1], duration=1.0)
    logger.info(f"鼠标返回原始位置: {original_pos}")
    
    logger.info("鼠标操作测试完成")


def test_find_image():
    """测试查找图片功能"""
    logger = get_logger(__name__)
    automation = AutomationCore()
    
    logger.info("测试查找图片功能...")
    
    # 这里需要有一个测试图片
    # 在实际使用中，您可以先使用capture_template.py创建一个测试模板
    test_template = project_root / "templates" / "test_template.png"
    
    if test_template.exists():
        logger.info(f"查找测试模板: {test_template}")
        
        # 查找图片
        location = automation.find_image(str(test_template))
        if location:
            logger.info(f"找到图片位置: {location}")
            
            # 获取中心点
            center = automation.find_image_center(str(test_template))
            if center:
                logger.info(f"图片中心点: {center}")
        else:
            logger.info("未找到测试模板")
    else:
        logger.warning("测试模板不存在，跳过查找测试")
        logger.info("请先运行 capture_template.py 创建一个测试模板")
    
    logger.info("查找图片测试完成")


def main():
    """主测试函数"""
    # 设置日志
    setup_logger(enable_file=False, enable_console=True)
    
    logger = get_logger(__name__)
    logger.info("开始基础功能测试...")
    
    try:
        # 测试截图功能
        test_screenshot()
        
        # 测试鼠标操作
        test_mouse_operations()
        
        # 测试查找图片
        test_find_image()
        
        logger.info("所有测试完成！")
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())