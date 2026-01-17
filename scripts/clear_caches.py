#!/usr/bin/env python3
"""
清理项目缓存
"""

import sys
import os
import importlib
from pathlib import Path
import shutil

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def clear_python_caches():
    """清除Python缓存"""
    print("🧹 清除Python缓存...")
    
    # 1. 清除 __pycache__ 目录
    for root, dirs, files in os.walk(project_root):
        if '__pycache__' in dirs:
            pycache_path = os.path.join(root, '__pycache__')
            shutil.rmtree(pycache_path, ignore_errors=True)
            print(f"   删除: {pycache_path}")
    
    # 2. 清除 .pyc 文件
    for root, dirs, files in os.walk(project_root):
        for file in files:
            if file.endswith('.pyc'):
                pyc_path = os.path.join(root, file)
                os.remove(pyc_path)
                print(f"   删除: {pyc_path}")
    
    # 3. 清除 import 缓存
    importlib.invalidate_caches()
    
    # 4. 清除特定模块
    modules_to_clear = [
        'pytesseract',
        'src.utils.config_manager',
        'src.core.smart_automation'
    ]
    
    for module in modules_to_clear:
        if module in sys.modules:
            del sys.modules[module]
            print(f"   清除模块缓存: {module}")
    
    print("✅ Python缓存已清除")

def clear_tesseract_cache():
    """清除Tesseract相关缓存"""
    print("\n🧹 清除Tesseract缓存...")
    
    try:
        import pytesseract
        
        # 清除tesseract_cmd属性
        if hasattr(pytesseract.pytesseract, 'tesseract_cmd'):
            delattr(pytesseract.pytesseract, 'tesseract_cmd')
            print("   清除tesseract_cmd属性")
        
        # 清除模块缓存
        if 'pytesseract' in sys.modules:
            del sys.modules['pytesseract']
        
        # 重新导入
        import pytesseract
        importlib.reload(pytesseract)
        
        print("✅ Tesseract缓存已清除")
        
    except ImportError:
        print("⚠️  pytesseract未安装")

def clear_output_caches():
    """清除输出目录缓存"""
    print("\n🧹 清除输出目录...")
    
    output_dirs = [
        project_root / "__pycache__",
        project_root / "build",
        project_root / "dist",
        project_root / ".pytest_cache",
        project_root / ".mypy_cache"
    ]
    
    for dir_path in output_dirs:
        if dir_path.exists():
            shutil.rmtree(dir_path, ignore_errors=True)
            print(f"   删除: {dir_path}")

def main():
    """主函数"""
    print("=" * 60)
    print("缓存清理工具")
    print("=" * 60)
    
    clear_python_caches()
    clear_tesseract_cache()
    clear_output_caches()
    
    print("\n" + "=" * 60)
    print("✅ 所有缓存已清除")
    print("提示: 重新运行你的程序以使配置生效")
    print("=" * 60)

if __name__ == "__main__":
    main()