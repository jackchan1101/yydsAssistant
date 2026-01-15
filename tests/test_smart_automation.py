#!/usr/bin/env python3
"""
测试智能自动化模块
"""

import sys
import os
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.smart_automation import SmartAutomation


def test_basic_functionality():
    """测试基本功能"""
    print("=== 测试智能自动化模块基本功能 ===")
    
    # 创建自动化实例
    automator = SmartAutomation()
    
    # 测试截图功能
    print("1. 测试截图功能...")
    screenshot = automator.capture_screen()
    print(f"   截图尺寸: {screenshot.shape}")
    
    # 测试预处理功能
    print("2. 测试图像预处理...")
    processed = automator.preprocess_image(screenshot)
    print(f"   处理后的图像尺寸: {processed.shape}")
    
    # 测试图像匹配 (需要准备一个模板图片)
    template_path = "templates/test_template.png"  # 这里放一个测试图片路径
    if os.path.exists(template_path):
        print(f"3. 测试图像匹配: {template_path}")
        result = automator.find_image(template_path)
        
        if result.found:
            print(f"   找到图像! 位置: {result.position}, 置信度: {result.confidence:.3f}")
            
            # 测试点击
            print("4. 测试图像点击...")
            success = automator.click_image(template_path)
            print(f"   点击结果: {'成功' if success else '失败'}")
        else:
            print(f"   未找到图像，最佳匹配值: {result.confidence:.3f}")
    else:
        print(f"3. 跳过图像匹配测试，模板图片不存在: {template_path}")
    
    # 测试查找所有匹配
    print("5. 测试查找所有匹配...")
    all_matches = automator.find_all_images(template_path) if os.path.exists(template_path) else []
    print(f"   找到 {len(all_matches)} 个匹配")
    
    print("=== 测试完成 ===")


def test_preprocessing_options():
    """测试不同的预处理选项"""
    print("\n=== 测试不同的预处理选项 ===")
    
    automator = SmartAutomation()
    screenshot = automator.capture_screen((0, 0, 800, 600))  # 截取部分屏幕
    
    # 测试不同的预处理配置
    configs = [
        {"grayscale": True, "blur_kernel": (5, 5)},
        {"grayscale": True, "blur_kernel": (0, 0), "threshold_type": "binary", "threshold_value": 127},
        {"grayscale": True, "blur_kernel": (3, 3), "threshold_type": "otsu"},
        {"grayscale": True, "blur_kernel": (5, 5), "canny": True, "canny_low": 50, "canny_high": 150},
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"  配置 {i}: {config}")
        processed = automator.preprocess_image(screenshot.copy(), config)
        print(f"    处理结果尺寸: {processed.shape}")
    
    print("=== 预处理测试完成 ===")


if __name__ == "__main__":
    print("智能自动化模块测试开始")
    print("=" * 50)
    
    try:
        test_basic_functionality()
        test_preprocessing_options()
        
        print("\n注意：要进行实际的图像匹配测试，需要在当前目录放置测试图片 'templates/test_template.png'")
        print("您可以截取屏幕上某个小图标（如Windows开始按钮）保存为 test_template.png 进行测试")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    input("\n按Enter键退出...")