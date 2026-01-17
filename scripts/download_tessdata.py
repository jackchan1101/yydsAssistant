#!/usr/bin/env python3
"""
下载Tesseract语言包
"""

import os
import urllib.request
from pathlib import Path

def download_tessdata():
    """下载语言包"""
    tessdata_dir = r"D:/DevTools/Tesseract-OCR/tessdata"
    
    # 确保目录存在
    os.makedirs(tessdata_dir, exist_ok=True)
    
    # 语言包下载地址
    languages = {
        'chi_sim': 'https://github.com/tesseract-ocr/tessdata_best/raw/main/chi_sim.traineddata',
        'chi_tra': 'https://github.com/tesseract-ocr/tessdata_best/raw/main/chi_tra.traineddata',
        'eng': 'https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata',
    }
    
    for lang, url in languages.items():
        file_path = os.path.join(tessdata_dir, f"{lang}.traineddata")
        
        if os.path.exists(file_path):
            print(f"✅ {lang}语言包已存在")
            continue
            
        print(f"⬇️  下载 {lang} 语言包...")
        try:
            urllib.request.urlretrieve(url, file_path)
            print(f"✅ {lang} 语言包下载完成")
        except Exception as e:
            print(f"❌ 下载 {lang} 失败: {e}")

if __name__ == "__main__":
    download_tessdata()