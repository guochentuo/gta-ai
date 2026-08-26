# OCR文字识别

本项目负责PP-OCRv6文字检测和识别。接口返回文字、坐标和置信度，业务数据持久化由
gta-ai之外的服务负责。

HTTP入口为`app.py`，运行服务为`gta-ai-ocr.service`。
