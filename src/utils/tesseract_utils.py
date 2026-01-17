#!/usr/bin/env python3
"""
Tesseract工具类 - 统一管理OCR功能
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# 配置日志
logger = logging.getLogger(__name__)

class TesseractConfig:
    """Tesseract配置管理器"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialized = True
            self._setup_tesseract()
    
    def _setup_tesseract(self):
        """设置Tesseract路径"""
        # 你的Tesseract安装路径
        tesseract_paths = [
            r"D:/DevTools/Tesseract-OCR/tesseract.exe",  # 你的安装路径
        ]
        
        found_path = None
        for path in tesseract_paths:
            if os.path.exists(path):
                found_path = path
                break
        
        if found_path:
            # 设置环境变量
            os.environ['TESSDATA_PREFIX'] = str(Path(found_path).parent / 'tessdata')
            
            # 设置pytesseract路径
            try:
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = found_path
                logger.info(f"✓ Tesseract路径已设置: {found_path}")
                
                # 验证设置
                self._verify_tesseract()
                
            except ImportError:
                logger.error("✗ 未安装pytesseract，请运行: pip install pytesseract")
            except Exception as e:
                logger.error(f"✗ 设置Tesseract失败: {e}")
        else:
            logger.warning("⚠️ 未找到Tesseract，OCR功能将不可用")
    
    def _verify_tesseract(self):
        """验证Tesseract设置"""
        try:
            import pytesseract
            version = pytesseract.get_tesseract_version()
            logger.info(f"✓ Tesseract版本: {version}")
            
            # 检查语言支持
            langs = pytesseract.get_languages(config='')
            logger.info(f"✓ 支持的语言: {langs}")
            
            if 'chi_sim' in langs:
                logger.info("✓ 简体中文语言包可用")
            else:
                logger.warning("⚠️ 简体中文语言包未找到")
                
        except Exception as e:
            logger.error(f"✗ Tesseract验证失败: {e}")
    
    def get_tesseract_cmd(self) -> str:
        """获取Tesseract命令路径"""
        try:
            import pytesseract
            return pytesseract.pytesseract.tesseract_cmd
        except:
            return ""
    
    def is_available(self) -> bool:
        """检查Tesseract是否可用"""
        cmd = self.get_tesseract_cmd()
        return cmd and os.path.exists(cmd)

# 创建全局实例
_tesseract_config = None

def get_tesseract_config() -> TesseractConfig:
    """获取Tesseract配置实例"""
    global _tesseract_config
    if _tesseract_config is None:
        _tesseract_config = TesseractConfig()
    return _tesseract_config

def init_tesseract():
    """初始化Tesseract（在项目启动时调用）"""
    return get_tesseract_config()