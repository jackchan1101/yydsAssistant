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
# 截屏
python scripts/capture_template.py

# 测试
python tests/test_basic.py
```

### 4. 集成OpenCV进行图像预处理
```bash
# 将要查找的图片保存到项目根目录 templates/test_template.png
python tests/test_smart_automation.py

# 创建模板
python scripts/create_template.py

# 运行综合测试
python tests/test_opencv_matching.py
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