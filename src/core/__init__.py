"""
核心功能模块
包含自动化工具的核心功能
"""

from .automation import AutomationCore, click, find_image, capture_screen
from .smart_automation import SmartAutomation, MatchResult

__all__ = ['Automation', 'SmartAutomation', 'MatchResult', 'click', 'find_image', 'capture_screen']