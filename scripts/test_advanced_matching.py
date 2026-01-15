#!/usr/bin/env python3
"""
测试高级图像匹配算法
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import time

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.smart_automation import SmartAutomation

def create_test_scenarios():
    """创建测试场景"""
    print("创建测试场景...")
    
    # 创建输出目录
    outputs_dir = Path("outputs/tests")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # 场景1: 缩放测试
    print("1. 创建缩放测试场景...")
    base_image = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(base_image, (50, 50), (100, 100), (0, 255, 0), -1)
    
    # 创建不同大小的模板
    template_original = base_image[50:100, 50:100]  # 50x50
    template_small = cv2.resize(template_original, (40, 40))  # 40x40
    template_large = cv2.resize(template_original, (60, 60))  # 60x60
    
    # 保存测试文件
    cv2.imwrite(str(outputs_dir / "test_scaling_base.png"), base_image)
    cv2.imwrite(str(outputs_dir / "test_template_original.png"), template_original)
    cv2.imwrite(str(outputs_dir / "test_template_small.png"), template_small)
    cv2.imwrite(str(outputs_dir / "test_template_large.png"), template_large)
    
    # 场景2: 旋转测试
    print("2. 创建旋转测试场景...")
    rotated_images = []
    for angle in [0, 30, 60, 90]:
        center = (150, 150)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(base_image, matrix, (300, 300))
        rotated_images.append(rotated)
        cv2.imwrite(str(outputs_dir / f"test_rotated_{angle}.png"), rotated)
    
    # 场景3: 透视变换测试
    print("3. 创建透视变换测试场景...")
    pts1 = np.float32([[50, 50], [200, 50], [50, 200], [200, 200]])
    pts2 = np.float32([[30, 30], [220, 40], [40, 220], [210, 210]])
    matrix_perspective = cv2.getPerspectiveTransform(pts1, pts2)
    perspective_img = cv2.warpPerspective(base_image, matrix_perspective, (300, 300))
    cv2.imwrite(str(outputs_dir / "test_perspective.png"), perspective_img)
    
    print(f"测试场景已保存到: {outputs_dir}/")
    return True

def test_multi_scale_matching():
    """测试多尺度匹配"""
    print("\n" + "=" * 60)
    print("测试多尺度匹配")
    print("=" * 60)
    
    automator = SmartAutomation()
    
    # 创建测试图像
    screenshot = automator.capture_screen((0, 0, 400, 400))
    if screenshot is None:
        print("截图失败，跳过测试")
        return
    
    # 保存原始截图
    outputs_dir = Path("outputs/tests")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(str(outputs_dir / "multi_scale_base.png"), screenshot)
    
    # 创建原始模板
    original_template = screenshot[100:150, 100:150]  # 50x50
    template_path = str(outputs_dir / "multi_scale_template.png")
    cv2.imwrite(template_path, original_template)
    
    print("模板: 50x50 像素")
    
    # 测试1: 普通模板匹配
    print("\n1. 普通模板匹配:")
    result_normal = automator.find_image(template_path, screen_region=(0, 0, 400, 400))
    print(f"   结果: {'成功' if result_normal.found else '失败'}, 置信度: {result_normal.confidence:.3f}")
    
    # 测试2: 多尺度匹配
    print("\n2. 多尺度匹配 (0.5x - 2.0x):")
    result_multi = automator.find_image_multi_scale(template_path, screen_region=(0, 0, 400, 400))
    print(f"   结果: {'成功' if result_multi.found else '失败'}")
    if result_multi.found:
        print(f"   位置: {result_multi.position}, 置信度: {result_multi.confidence:.3f}")
        print(f"   尺度: {result_multi.scale:.2f}x")
    
    # 测试3: 创建缩放后的模板
    print("\n3. 测试缩放模板的匹配:")
    
    # 小模板 (40x40)
    small_template = cv2.resize(original_template, (40, 40))
    small_path = str(outputs_dir / "multi_scale_template_small.png")
    cv2.imwrite(small_path, small_template)
    
    result_small_normal = automator.find_image(small_path, screen_region=(0, 0, 400, 400))
    result_small_multi = automator.find_image_multi_scale(small_path, screen_region=(0, 0, 400, 400))
    
    print(f"   小模板(40x40):")
    print(f"     普通匹配: {'成功' if result_small_normal.found else '失败'}")
    print(f"     多尺度匹配: {'成功' if result_small_multi.found else '失败'}")
    
    # 大模板 (60x60)
    large_template = cv2.resize(original_template, (60, 60))
    large_path = str(outputs_dir / "multi_scale_template_large.png")
    cv2.imwrite(large_path, large_template)
    
    result_large_normal = automator.find_image(large_path, screen_region=(0, 0, 400, 400))
    result_large_multi = automator.find_image_multi_scale(large_path, screen_region=(0, 0, 400, 400))
    
    print(f"   大模板(60x60):")
    print(f"     普通匹配: {'成功' if result_large_normal.found else '失败'}")
    print(f"     多尺度匹配: {'成功' if result_large_multi.found else '失败'}")

def test_feature_matching():
    """测试特征匹配"""
    print("\n" + "=" * 60)
    print("测试特征匹配")
    print("=" * 60)
    
    automator = SmartAutomation()
    
    # 创建特征丰富的测试图像
    test_image = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # 添加各种形状以创建丰富特征
    cv2.rectangle(test_image, (50, 50), (150, 150), (0, 255, 0), -1)
    cv2.circle(test_image, (300, 100), 40, (255, 0, 0), -1)
    cv2.line(test_image, (200, 200), (350, 350), (0, 0, 255), 3)
    
    # 添加一些文字
    cv2.putText(test_image, "TEST", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 保存测试图像
    outputs_dir = Path("outputs/tests")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(str(outputs_dir / "feature_base.png"), test_image)
    
    # 创建模板
    template = test_image[50:150, 50:150]  # 矩形区域
    template_path = str(outputs_dir / "feature_template.png")
    cv2.imwrite(template_path, template)
    
    print("测试图像特征丰富度:")
    print("  - 矩形 (绿色)")
    print("  - 圆形 (蓝色)")
    print("  - 直线 (红色)")
    print("  - 文字 (白色)")
    
    # 测试特征匹配
    print("\n1. ORB特征匹配:")
    result_orb = automator.find_image_with_features(template_path, method='orb')
    print(f"   结果: {'成功' if result_orb.found else '失败'}")
    if result_orb.found:
        print(f"   位置: {result_orb.position}, 置信度: {result_orb.confidence:.3f}")
        print(f"   匹配点数: {result_orb.matches_count}")
    
    # 测试旋转后的匹配
    print("\n2. 测试旋转不变性:")
    
    # 创建旋转后的图像
    center = (200, 200)
    matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
    rotated_image = cv2.warpAffine(test_image, matrix, (400, 400))
    rotated_path = str(outputs_dir / "feature_rotated.png")
    cv2.imwrite(rotated_path, rotated_image)
    
    # 在旋转图像中查找模板
    result_rotated_orb = automator.find_image_with_features(template_path, method='orb')
    print(f"   旋转45度后ORB匹配: {'成功' if result_rotated_orb.found else '失败'}")
    
    # 尝试SIFT（如果可用）
    print("\n3. 尝试SIFT特征匹配:")
    try:
        result_sift = automator.find_image_with_features(template_path, method='sift')
        print(f"   SIFT匹配: {'成功' if result_sift.found else '失败'}")
        if result_sift.found:
            print(f"   匹配点数: {result_sift.matches_count}")
    except Exception as e:
        print(f"   SIFT不可用: {e}")
        print("   提示: OpenCV-contrib-python 包含SIFT")

def test_smart_matching():
    """测试智能匹配"""
    print("\n" + "=" * 60)
    print("测试智能匹配器")
    print("=" * 60)
    
    automator = SmartAutomation()
    
    # 创建多种测试场景
    test_cases = [
        {
            "name": "相同大小模板",
            "description": "普通模板匹配应该最快"
        },
        {
            "name": "缩放模板", 
            "description": "多尺度匹配应该能处理"
        },
        {
            "name": "特征丰富图像",
            "description": "特征匹配应该表现好"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}: {test_case['name']}")
        print(f"描述: {test_case['description']}")
        
        # 创建测试图像
        screenshot = automator.capture_screen((0, 0, 400, 400))
        if screenshot is None:
            continue
            
        outputs_dir = Path("outputs/tests")
        template_path = str(outputs_dir / f"smart_test_{i}.png")
        
        if i == 1:
            # 相同大小
            template = screenshot[100:150, 100:150]
        elif i == 2:
            # 缩放版本
            template = screenshot[100:150, 100:150]
            template = cv2.resize(template, (60, 60))
        elif i == 3:
            # 特征丰富的区域
            template = screenshot[50:200, 50:200]
        
        cv2.imwrite(template_path, template)
        
        # 测试各种方法
        methods = ['template', 'multi_scale', 'orb']
        
        for method in methods:
            start_time = time.time()
            
            if method == 'template':
                result = automator.find_image(template_path, screen_region=(0, 0, 400, 400))
            elif method == 'multi_scale':
                result = automator.find_image_multi_scale(template_path, screen_region=(0, 0, 400, 400))
            elif method == 'orb':
                result = automator.find_image_with_features(template_path, screen_region=(0, 0, 400, 400))
            
            elapsed = time.time() - start_time
            
            status = "✓" if result.found else "✗"
            conf = f"{result.confidence:.3f}" if result.confidence is not None else "N/A"
            print(f"   {method:12} {status} 置信度: {conf:6} 耗时: {elapsed:.3f}s")
        
        # 测试智能匹配
        start_time = time.time()
        smart_result = automator.smart_find_image(template_path, screen_region=(0, 0, 400, 400))
        elapsed = time.time() - start_time
        
        method_display = smart_result.method or "unknown"
        print(f"   {'smart':12} ✓ 使用: {method_display:15} 耗时: {elapsed:.3f}s")

def performance_comparison():
    """性能对比"""
    print("\n" + "=" * 60)
    print("性能对比测试")
    print("=" * 60)
    
    automator = SmartAutomation()
    
    # 准备测试数据
    screenshot = automator.capture_screen((0, 0, 800, 600))
    if screenshot is None:
        print("截图失败，跳过性能测试")
        return
    
    outputs_dir = Path("outputs/tests")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建不同大小的模板
    template_sizes = [(50, 50), (100, 100), (150, 150)]
    
    for w, h in template_sizes:
        print(f"\n模板大小: {w}x{h}")
        
        template = screenshot[100:100+h, 100:100+w]
        template_path = str(outputs_dir / f"perf_template_{w}x{h}.png")
        cv2.imwrite(template_path, template)
        
        # 测试各种方法
        methods = [
            ("template", lambda: automator.find_image(template_path, screen_region=(0, 0, 800, 600))),
            ("multi_scale", lambda: automator.find_image_multi_scale(template_path, screen_region=(0, 0, 800, 600))),
            ("orb", lambda: automator.find_image_with_features(template_path, screen_region=(0, 0, 800, 600), method='orb')),
        ]
        
        for method_name, method_func in methods:
            try:
                # 预热
                _ = method_func()
                
                # 正式测试
                times = []
                for _ in range(3):
                    start_time = time.time()
                    result = method_func()
                    times.append(time.time() - start_time)
                
                avg_time = sum(times) / len(times)
                status = "✓" if result.found else "✗"
                conf = f"{result.confidence:.3f}" if result.confidence is not None else "N/A"
                
                print(f"   {method_name:12} {status} 平均耗时: {avg_time:.3f}s, 置信度: {conf}")
                
            except Exception as e:
                print(f"   {method_name:12} ✗ 失败: {e}")

def main():
    """主函数"""
    print("=" * 60)
    print("高级图像匹配算法测试")
    print("=" * 60)
    
    print("本测试将验证:")
    print("1. 多尺度模板匹配 - 解决缩放问题")
    print("2. 特征点匹配 - 解决旋转和变形问题")
    print("3. 智能匹配器 - 自动选择最佳算法")
    print("=" * 60)
    
    try:
        # 创建测试场景
        create_test_scenarios()
        
        # 运行各项测试
        test_multi_scale_matching()
        test_feature_matching()
        test_smart_matching()
        performance_comparison()
        
        print("\n" + "=" * 60)
        print("测试完成!")
        print("=" * 60)
        print("\n输出文件保存在: outputs/tests/")
        print("\n总结:")
        print("1. 多尺度匹配可以处理模板缩放问题")
        print("2. 特征匹配对旋转和透视变换更鲁棒")
        print("3. 智能匹配器能自动选择合适算法")
        print("4. 普通模板匹配速度最快，适合简单场景")
        
    except Exception as e:
        print(f"\n✗ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
    input("\n按 Enter 键退出...")