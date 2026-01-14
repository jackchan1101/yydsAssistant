# 自动化工具

基于Python开发的自动化工具，支持屏幕识别和自动点击功能。

## 功能特性

### 第一阶段：基础自动化实现
- ✅ 屏幕截图功能
- ✅ 基础图像匹配
- ✅ 鼠标点击控制
- ✅ 日志记录系统
- ✅ 配置文件管理
- 🔄 模板图片捕获工具

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行主程序
```bash
python main.py
```

### 3. 捕获模板图片
```bash
python capture_template.py
```

### 4. 运行测试
```bash
python tests/test_basic.py
```

### 基本使用
```python
from src.core.automation import create_automation

创建自动化实例

automation = create_automation()

截图

screenshot = automation.screenshot()

查找并点击图片

automation.click_image("templates/button.png")

等待图片出现

center = automation.wait_until_image_appears("templates/dialog.png", timeout=10)
```

```bash

```