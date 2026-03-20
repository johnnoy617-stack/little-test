# Flask RAG Question Bot Demo

一个面向 PythonAnywhere 免费版的演示型知识库问答机器人。管理员可通过无密码的 `/admin` 后台上传 PDF，系统会同步解析文档、切分文本、构建轻量 TF-IDF 检索索引，并在用户提问时调用 SiliconFlow 的 OpenAI 兼容接口生成回答。

## 功能

- PDF 上传、删除、同步重建索引
- 基于 `pypdf` 的文本型 PDF 解析
- 基于 `TfidfVectorizer` 的轻量 RAG 检索
- 基于 SiliconFlow API 的答案生成
- 服务端渲染的问答页与演示后台

## 技术栈

- Flask + Jinja2 + SQLAlchemy + SQLite
- pypdf
- scikit-learn / scipy / joblib
- requests

## 本地启动

1. 创建虚拟环境并安装依赖

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. 配置环境变量

复制 `.env.example` 为 `.env`，至少填入：

```env
SECRET_KEY=replace-me
SILICONFLOW_API_KEY=your-api-key
SILICONFLOW_MODEL=Qwen/Qwen2.5-7B-Instruct
```

3. 启动应用

```bash
flask --app wsgi:app run --debug
```

访问：

- 首页：`http://127.0.0.1:5000/`
- 后台：`http://127.0.0.1:5000/admin`

## 环境变量

- `SECRET_KEY`：Flask session 密钥
- `DATABASE_URL`：数据库地址，默认 `sqlite:///instance/app.db`
- `MAX_CONTENT_LENGTH_MB`：上传大小限制，默认 `10`
- `CHUNK_SIZE`：分块长度，默认 `700`
- `CHUNK_OVERLAP`：分块重叠，默认 `120`
- `TOP_K`：检索召回数量，默认 `5`
- `SCORE_THRESHOLD`：最低可回答阈值，默认 `0.08`
- `SILICONFLOW_API_KEY`：SiliconFlow API Key
- `SILICONFLOW_BASE_URL`：默认 `https://api.siliconflow.cn/v1`
- `SILICONFLOW_MODEL`：默认 `Qwen/Qwen2.5-7B-Instruct`
- `REQUEST_TIMEOUT`：模型请求超时秒数，默认 `45`

## PythonAnywhere 部署

1. 在 PythonAnywhere 创建新的 Flask web app（Python 3.10+）。
2. 将项目上传到你的 home 目录，例如 `/home/yourname/questionbot`。
3. 创建虚拟环境并安装依赖：

```bash
mkvirtualenv --python=/usr/bin/python3.10 questionbot-env
pip install -r /home/yourname/questionbot/requirements.txt
```

4. 在 web app 的 WSGI 配置文件中指向项目：

```python
import sys
path = '/home/yourname/questionbot'
if path not in sys.path:
    sys.path.append(path)

from wsgi import app as application
```

5. 在 PythonAnywhere 的 Web 页面配置环境变量：

- `SECRET_KEY`
- `SILICONFLOW_API_KEY`
- `SILICONFLOW_MODEL`

6. Reload web app。

应用会自动在 `instance/` 下创建数据库、上传目录和索引目录。

## 注意事项

- `/admin` 没有密码，仅适合演示。
- 一期仅支持文本型 PDF，不支持扫描件 OCR。
- 若知识库为空或模型 API 未配置，首页会给出明确提示。
