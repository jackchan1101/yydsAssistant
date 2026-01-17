# 自动化工具

基于Python开发的自动化工具，支持屏幕识别和自动点击功能。

❌ 表示未完成/错误
⬜ 表示未开始
🔄 表示进行中
⏳ 表示等待中

## 功能特性

### 第一阶段：基础自动化实现
- ✅ 屏幕截图功能
- ✅ 基础图像匹配
- ✅ 鼠标点击控制
- ✅ 日志记录系统
- ✅ 配置文件管理
- ✅ 模板图片捕获工具

#### 1. 安装依赖
```bash
pip install -r requirements.txt
```

#### 2. 运行主程序
```bash
python main.py
```

#### 3. 捕获模板图片
```bash
# 截屏
python scripts/capture_template.py

# 测试
python tests/test_basic.py
```

### 第二阶段：基础自动化实现
- ✅ 集成OpenCV进行图像预处理
- ✅ 实现多种图像匹配算法
- ✅ 集成Tesseract OCR识别文本
- ✅ 开发自适应阈值识别系统
- ⬜ 添加图像特征点检测功能

#### 1. 集成OpenCV进行图像预处理
```bash
# 将要查找的图片保存到项目templates目录
python scripts/test_smart_automation.py

# 创建模板
python scripts/create_template.py

# 运行综合测试
python scripts/test_opencv_matching.py
```

#### 2. 实现多种图像匹配算法

```bash
# 高级图像匹配测试   普通模板匹配、多尺度匹配、特征匹配、智能匹配器、性能表现
python scripts/test_advanced_matching.py

# 在屏幕上实际查找元素测试   Windows UI元素查找、桌面图标查找、网页浏览器元素查找、自定义模板匹配、性能优化测试
python scripts/test_real_world.py
```

#### 3. 集成Tesseract OCR识别文本
```bash
# 没有语言包先下载
python scripts/download_tessdata.py

# OCR测试
python scripts/test_ocr_recognition.py
```

#### 4. 开发自适应阈值识别系统
```bash
# 自适应阈值识别测试
python scripts/test_adaptive_threshold.py
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
