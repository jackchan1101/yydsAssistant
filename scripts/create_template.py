#!/usr/bin/env python3
"""
创建测试模板
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.smart_automation import SmartAutomation

def create_test_template():
    """创建测试模板"""
    print("=" * 60)
    print("创建测试模板")
    print("=" * 60)
    
    # 创建templates目录
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    
    # 初始化自动化
    automator = SmartAutomation()
    
    print("1. 截取屏幕...")
    screenshot = automator.capture_screen()
    print(f"   截图尺寸: {screenshot.shape}")
    
    # 保存完整截图
    cv2.imwrite("outputs/full_screenshot.png", screenshot)
    print("   完整截图已保存为: outputs/full_screenshot.png")
    
    # 从屏幕中心创建一个测试模板
    print("\n2. 创建测试模板...")
    height, width = screenshot.shape[:2]
    
    # 屏幕中心区域
    center_x, center_y = width // 2, height // 2
    template_size = 100  # 100x100的模板
    
    # 确保坐标在范围内
    start_x = max(0, center_x - template_size // 2)
    start_y = max(0, center_y - template_size // 2)
    end_x = min(width, start_x + template_size)
    end_y = min(height, start_y + template_size)
    
    # 截取模板
    template = screenshot[start_y:end_y, start_x:end_x]
    
    if template.size > 0:
        template_path = templates_dir / "test_template.png"
        cv2.imwrite(str(template_path), template)
        print(f"✓ 测试模板创建成功: {template_path}")
        print(f"   模板位置: ({start_x}, {start_y}) 到 ({end_x}, {end_y})")
        print(f"   模板尺寸: {template.shape}")
        
        # 在完整截图上标记模板位置
        marked_screenshot = screenshot.copy()
        cv2.rectangle(marked_screenshot, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
        cv2.imwrite("outputs/marked_screenshot.png", marked_screenshot)
        print("   标记模板位置的截图已保存为: outputs/marked_screenshot.png")
        
        return True, str(template_path)
    else:
        print("✗ 无法创建模板")
        return False, None

def test_template_matching(template_path):
    """测试模板匹配"""
    print("\n3. 测试模板匹配...")
    
    automator = SmartAutomation()
    
    # 测试1: 正常匹配
    print("   测试1: 正常匹配...")
    result = automator.find_image(template_path)
    
    if result.found:
        print(f"   ✓ 匹配成功!")
        print(f"     位置: {result.position}")
        print(f"     置信度: {result.confidence:.3f}")
        
        # 验证位置是否正确
        template_img = cv2.imread(template_path)
        h, w = template_img.shape[:2]
        
        # 计算预期位置（模板中心点）
        expected_x = (w // 2) + 50  # 模板是从(50, 50)开始的
        expected_y = (h // 2) + 50
        
        print(f"     预期位置大约在: ({expected_x}, {expected_y})")
    else:
        print(f"   ✗ 未找到匹配，置信度: {result.confidence:.3f}")
    
    # 测试2: 使用不同的阈值
    print("\n   测试2: 使用高阈值(0.95)...")
    result2 = automator.find_image(template_path, threshold=0.95)
    print(f"     结果: {'找到' if result2.found else '未找到'}，置信度: {result2.confidence:.3f}")
    
    # 测试3: 使用低阈值(0.5)
    print("\n   测试3: 使用低阈值(0.5)...")
    result3 = automator.find_image(template_path, threshold=0.5)
    print(f"     结果: {'找到' if result3.found else '未找到'}，置信度: {result3.confidence:.3f}")
    
    # 测试4: 测试点击功能
    print("\n   测试4: 测试点击功能...")
    print("   注意: 鼠标将移动到模板位置")
    success = automator.click_image(template_path, offset=(0, 0))
    print(f"     点击结果: {'成功' if success else '失败'}")

def test_different_matching_methods(template_path):
    """测试不同的匹配方法"""
    print("\n4. 测试不同的匹配方法...")
    
    methods = [
        ("CCOEFF_NORMED", cv2.TM_CCOEFF_NORMED),
        ("CCORR_NORMED", cv2.TM_CCORR_NORMED),
        ("SQDIFF_NORMED", cv2.TM_SQDIFF_NORMED),
        ("CCOEFF", cv2.TM_CCOEFF),
        ("CCORR", cv2.TM_CCORR),
        ("SQDIFF", cv2.TM_SQDIFF),
    ]
    
    for method_name, method_code in methods:
        automator = SmartAutomation({
            'matching': {
                'method': method_code,
                'threshold': 0.8
            }
        })
        
        result = automator.find_image(template_path)
        status = "找到" if result.found else "未找到"
        print(f"   {method_name:15} {status:6} 置信度: {result.confidence:.3f}")

def main():
    """主函数"""
    print("智能自动化模块 - 图像匹配测试")
    print("=" * 60)
    
    try:
        # 1. 创建测试模板
        success, template_path = create_test_template()
        
        if not success or not template_path:
            print("无法创建模板，测试终止")
            return
        
        # 2. 测试模板匹配
        test_template_matching(template_path)
        
        # 3. 测试不同的匹配方法
        test_different_matching_methods(template_path)
        
        print("\n" + "=" * 60)
        print("测试完成！生成的文件：")
        print("=" * 60)
        
        for file in ["outputs/full_screenshot.png", "outputs/marked_screenshot.png", "templates/test_template.png"]:
            if os.path.exists(file):
                print(f"  - {file}")
        
        print("\n下一步：")
        print("1. 打开 outputs/marked_screenshot.png 查看模板位置")
        print("2. 运行测试查看匹配结果")
        print("3. 可以修改模板或测试不同的匹配参数")
        
    except Exception as e:
        print(f"\n✗ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
    input("\n按 Enter 键退出...")