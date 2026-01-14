"""
配置管理器模块，负责加载和管理配置文件
"""
import yaml
import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
from .logger import get_logger


logger = get_logger(__name__)


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: str = "./configs"):
        """
        初始化配置管理器
        
        Args:
            config_dir: 配置文件目录
        """
        self.config_dir = Path(config_dir)
        self.configs: Dict[str, Any] = {}
        self.configs_cache: Dict[str, Any] = {}
        
        # 确保配置目录存在
        os.makedirs(self.config_dir, exist_ok=True)
        
        logger.info(f"配置管理器初始化，配置目录: {self.config_dir}")
    
    def load_config(self, filename: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            filename: 配置文件名
            use_cache: 是否使用缓存
            
        Returns:
            配置字典
        """
        if use_cache and filename in self.configs_cache:
            logger.debug(f"从缓存加载配置: {filename}")
            return self.configs_cache[filename]
        
        config_path = self.config_dir / filename
        config_data = {}
        
        if not config_path.exists():
            logger.warning(f"配置文件不存在: {config_path}")
            return config_data
        
        try:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f) or {}
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            else:
                logger.error(f"不支持的配置文件格式: {config_path.suffix}")
                return config_data
            
            self.configs_cache[filename] = config_data
            logger.info(f"配置文件加载成功: {filename}")
            
        except Exception as e:
            logger.error(f"加载配置文件失败 {filename}: {e}")
        
        return config_data
    
    def save_config(self, filename: str, config_data: Dict[str, Any]) -> bool:
        """
        保存配置文件
        
        Args:
            filename: 配置文件名
            config_data: 配置数据
            
        Returns:
            是否保存成功
        """
        config_path = self.config_dir / filename
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
                elif config_path.suffix.lower() == '.json':
                    json.dump(config_data, f, ensure_ascii=False, indent=2)
                else:
                    logger.error(f"不支持的配置文件格式: {config_path.suffix}")
                    return False
            
            # 更新缓存
            self.configs_cache[filename] = config_data
            logger.info(f"配置文件保存成功: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"保存配置文件失败 {filename}: {e}")
            return False
    
    def get_value(self, 
                  filename: str, 
                  key_path: str, 
                  default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            filename: 配置文件名
            key_path: 键路径，用点分隔
            default: 默认值
            
        Returns:
            配置值
        """
        config_data = self.load_config(filename)
        
        # 按路径获取值
        value = config_data
        for key in key_path.split('.'):
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set_value(self, 
                  filename: str, 
                  key_path: str, 
                  value: Any) -> bool:
        """
        设置配置值
        
        Args:
            filename: 配置文件名
            key_path: 键路径，用点分隔
            value: 要设置的值
            
        Returns:
            是否设置成功
        """
        config_data = self.load_config(filename)
        
        # 创建或更新配置路径
        keys = key_path.split('.')
        current = config_data
        
        for i, key in enumerate(keys[:-1]):
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
        
        # 保存更新后的配置
        return self.save_config(filename, config_data)


# 全局配置管理器实例
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_dir: str = "./configs") -> ConfigManager:
    """
    获取全局配置管理器实例
    
    Args:
        config_dir: 配置目录
        
    Returns:
        配置管理器实例
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_dir)
    return _config_manager


# 便捷函数
def load_config(filename: str) -> Dict[str, Any]:
    """加载配置文件"""
    return get_config_manager().load_config(filename)


def save_config(filename: str, config_data: Dict[str, Any]) -> bool:
    """保存配置文件"""
    return get_config_manager().save_config(filename, config_data)


def get_config_value(filename: str, key_path: str, default: Any = None) -> Any:
    """获取配置值"""
    return get_config_manager().get_value(filename, key_path, default)


def set_config_value(filename: str, key_path: str, value: Any) -> bool:
    """设置配置值"""
    return get_config_manager().set_value(filename, key_path, value)