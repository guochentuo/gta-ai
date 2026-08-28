from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "router"))

from runtime_logging import configure, error, info, shutdown


def main() -> None:
    configure("gta-ai-vllm")
    info("27B模型进程正在启动", console=True)
    for line in sys.stdin:
        message = line.rstrip()
        if not message:
            continue
        lowered = message.lower()
        if "error" in lowered or "traceback" in lowered or "exception" in lowered:
            error(message, source="vllm")
        else:
            info(message, source="vllm")
    info("27B模型已停止", console=True)
    shutdown()


if __name__ == "__main__":
    main()
