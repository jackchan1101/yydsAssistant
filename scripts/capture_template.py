#!/usr/bin/env python3
"""
模板图片捕获工具
用于捕获界面元素作为模板图片
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
import pyautogui
from PIL import Image
from src.utils.logger import setup_logger, get_logger


def capture_template():
    """捕获模板图片"""
    logger = get_logger(__name__)
    
    # 创建模板目录
    template_dir = project_root / "templates"
    template_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*50)
    print("模板图片捕获工具")
    print("="*50)
    print("使用说明:")
    print("1. 将鼠标移动到您想要捕获的区域")
    print("2. 按 Enter 键开始选择区域")
    print("3. 拖动鼠标选择区域")
    print("4. 输入模板名称（不要带扩展名）")
    print("5. 按 q 退出")
    print("="*50)
    
    while True:
        try:
            input("\n将鼠标移动到目标区域，然后按 Enter 键开始选择 (q退出): ")
            
            if input().lower() == 'q':
                print("退出捕获工具")
                break
            
            print("5秒后将开始捕获，请准备...")
            time.sleep(5)
            
            # 获取鼠标位置
            start_x, start_y = pyautogui.position()
            print(f"起始位置: ({start_x}, {start_y})")
            
            print("移动鼠标到结束位置，然后按 Enter 键...")
            input()
            
            end_x, end_y = pyautogui.position()
            print(f"结束位置: ({end_x}, {end_y})")
            
            # 计算区域
            left = min(start_x, end_x)
            top = min(start_y, end_y)
            width = abs(end_x - start_x)
            height = abs(end_y - start_y)
            
            print(f"捕获区域: ({left}, {top}, {width}, {height})")
            
            # 获取模板名称
            template_name = input("请输入模板名称: ").strip()
            if not template_name:
                print("模板名称不能为空")
                continue
            
            # 截图
            screenshot = pyautogui.screenshot(region=(left, top, width, height))
            
            # 保存模板
            template_path = template_dir / f"{template_name}.png"
            screenshot.save(template_path)
            
            print(f"模板已保存: {template_path}")
            
            # 预览
            # screenshot.show()
            
        except KeyboardInterrupt:
            print("\n捕获中断")
            break
        except Exception as e:
            print(f"捕获失败: {e}")
            continue


if __name__ == "__main__":
    # 设置日志
    setup_logger(enable_file=False, enable_console=True)
    
    # 运行捕获工具
    capture_template()