"""容器入口：初始化 SQLite 表结构后启动 waitress。"""

from waitress import serve

from app import app

app.init_db()
serve(app.app, host="0.0.0.0", port=8000)
