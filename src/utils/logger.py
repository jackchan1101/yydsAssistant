"""
日志工具模块，提供统一的日志记录功能
"""
import logging
import logging.handlers
import os
from datetime import datetime
from typing import Optional


class GameAutomationLogger:
    """游戏自动化日志记录器"""
    
    def __init__(self, 
                 name: str = "GameAutomation",
                 log_level: str = "INFO",
                 enable_file: bool = True,
                 enable_console: bool = True,
                 log_dir: str = "./logs"):
        """
        初始化日志记录器
        
        Args:
            name: 日志器名称
            log_level: 日志级别
            enable_file: 是否启用文件日志
            enable_console: 是否启用控制台日志
            log_dir: 日志目录
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # 清除已有处理器
        self.logger.handlers.clear()
        
        # 设置日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台处理器
        if enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # 文件处理器
        if enable_file:
            # 确保日志目录存在
            os.makedirs(log_dir, exist_ok=True)
            
            # 创建带时间戳的日志文件名
            log_file = os.path.join(
                log_dir, 
                f"game_automation_{datetime.now().strftime('%Y%m%d')}.log"
            )
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def get_logger(self) -> logging.Logger:
        """获取日志记录器实例"""
        return self.logger


# 全局日志实例
_logger_instance: Optional[GameAutomationLogger] = None


def setup_logger(**kwargs) -> logging.Logger:
    """
    设置全局日志记录器
    
    Returns:
        配置好的日志记录器
    """
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = GameAutomationLogger(**kwargs)
    return _logger_instance.get_logger()


def get_logger(name: str = "GameAutomation") -> logging.Logger:
    """
    获取日志记录器
    
    Args:
        name: 日志器名称
        
    Returns:
        日志记录器实例
    """
    if _logger_instance is None:
        return setup_logger(name=name)
    return _logger_instance.get_logger()


# 导出便捷函数
info = lambda msg, *args: get_logger().info(msg, *args)
debug = lambda msg, *args: get_logger().debug(msg, *args)
warning = lambda msg, *args: get_logger().warning(msg, *args)
error = lambda msg, *args: get_logger().error(msg, *args)
critical = lambda msg, *args: get_logger().critical(msg, *args)