#!/usr/bin/env python3
"""
自动化工具 - 主程序入口
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger, get_logger
from src.utils.config_manager import get_config_manager
from src.core.automation import AutomationCore, test_basic_functions


def initialize_environment():
    """初始化环境"""
    # 创建必要的目录
    directories = [
        "logs",
        "screenshots",
        "templates",
        "data",
        "configs"
    ]
    
    for dir_name in directories:
        dir_path = project_root / dir_name
        dir_path.mkdir(exist_ok=True)

    # 初始化Tesseract配置
    from src.utils.tesseract_utils import init_tesseract
    tesseract_config = init_tesseract()

    # 检查状态
    if tesseract_config.is_available():
        print("✓ Tesseract配置成功")
        cmd = tesseract_config.get_tesseract_cmd()
        print(f"  Tesseract路径: {cmd}")
    else:
        print("✗ Tesseract配置失败")
    
    # 设置日志
    logger = setup_logger(
        name="GameAutomation",
        log_level="INFO",
        enable_file=True,
        enable_console=True,
        log_dir=str(project_root / "logs")
    )
    
    logger.info("=" * 50)
    logger.info("自动化工具启动")
    logger.info(f"项目根目录: {project_root}")
    logger.info("=" * 50)
    
    return logger


def main():
    """主函数"""
    # 初始化环境
    logger = initialize_environment()
    
    try:
        # 加载配置
        config_manager = get_config_manager(str(project_root / "configs"))
        
        # 检查默认配置文件是否存在
        settings_path = project_root / "configs" / "settings.yaml"
        if not settings_path.exists():
            logger.warning("未找到配置文件，将创建默认配置...")
            # 这里可以添加默认配置的创建逻辑
            
        # 创建自动化核心实例
        automation = AutomationCore(str(project_root / "configs"))
        
        logger.info("自动化工具初始化完成")
        logger.info("按 Ctrl+C 退出程序")
        
        # 保持程序运行
        while True:
            command = input("\n请输入命令 (test/basic/exit): ").strip().lower()
            
            if command == "test":
                logger.info("开始基础功能测试...")
                test_basic_functions()
                
            elif command == "basic":
                logger.info("基础功能测试模式...")
                # 这里可以添加更多测试功能
                
            elif command == "exit":
                logger.info("正在退出程序...")
                break
                
            else:
                logger.info("可用命令: test, basic, exit")
                
    except KeyboardInterrupt:
        logger.info("\n收到退出信号，程序终止")
    except Exception as e:
        logger.error(f"程序运行异常: {e}", exc_info=True)
    finally:
        logger.info("自动化工具关闭")


if __name__ == "__main__":
    main()