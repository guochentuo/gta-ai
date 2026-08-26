# ASR语音识别

本项目负责faster-whisper-large-v3语音转文字、时间轴、语音门控和健康状态，不选择媒体
存储路径，也不持久化worker业务结果。

HTTP入口为`app.py`，运行服务为`gta-ai-asr.service`。
