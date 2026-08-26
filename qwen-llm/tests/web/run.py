from __future__ import annotations

import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI

PROJECT_ROOT = Path(__file__).resolve().parents[2]

for source_path in (PROJECT_ROOT,):
    path_text = str(source_path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from tests.web.app import create_web_router  # noqa: E402

WEB_HOST = "0.0.0.0"
WEB_PORT = 8080


def create_app() -> FastAPI:
    app = FastAPI(title="GTA AI 测试页面")
    app.include_router(create_web_router())
    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(app, host=WEB_HOST, port=WEB_PORT)
