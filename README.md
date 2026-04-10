# HKU Knowledge Companion

一个面向课程设计演示的知识库问答网站。管理员可以通过 `/admin` 上传 PDF 或 DOCX 文档，系统会同步完成文本解析、分块、向量化并写入 Qdrant Cloud，然后基于检索结果与现有大模型 API 输出答案。

## 当前功能

- PDF / DOCX 上传与删除
- 基于 `pypdf` 和 `python-docx` 的文本解析
- 基于外部 Embedding API 的向量化
- 基于 Qdrant Cloud 的向量检索
- 可选 rerank 接口
- 基于现有问答模型 API 的答案生成
- HKU 风格的课程设计演示首页与后台

## 运行环境

- Python 3.10+
- Flask + SQLAlchemy + SQLite
- Qdrant Cloud Free
- 外部大模型 / embedding / rerank API

## 本地启动

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 配置环境变量

复制 `.env.example` 为 `.env`，至少填写：

```env
SECRET_KEY=replace-me
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your-qdrant-key
QDRANT_COLLECTION=course_kb
SILICONFLOW_API_KEY=your-api-key
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
EMBEDDING_MODEL=BAAI/bge-m3
```

如果 embedding / rerank / LLM 使用同一家 API，可以只配置 `SILICONFLOW_API_KEY` 和默认 base URL；其余项会自动回退复用。

3. 启动应用

```bash
flask --app wsgi:app run --debug
```

访问：

- 首页：`http://127.0.0.1:5000/`
- 后台：`http://127.0.0.1:5000/admin`

## 关键环境变量

- `SECRET_KEY`
- `DATABASE_URL`，默认 `sqlite:///instance/app.db`
- `UPLOAD_DIR`
- `MAX_CONTENT_LENGTH_MB`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `TOP_K_INITIAL`
- `TOP_K_FINAL`
- `VECTOR_SCORE_THRESHOLD`
- `QDRANT_URL`
- `QDRANT_API_KEY`
- `QDRANT_COLLECTION`
- `SILICONFLOW_API_KEY`
- `SILICONFLOW_BASE_URL`
- `LLM_API_KEY`
- `LLM_BASE_URL`
- `LLM_MODEL`
- `EMBEDDING_API_KEY`
- `EMBEDDING_BASE_URL`
- `EMBEDDING_MODEL`
- `RERANK_ENABLED`
- `RERANK_API_KEY`
- `RERANK_BASE_URL`
- `RERANK_MODEL`
- `REQUEST_TIMEOUT`
- `SITE_TITLE`
- `SITE_SUBTITLE`

## 腾讯云 Lighthouse 部署

### 1. 推荐服务器配置

- 轻量应用服务器 Lighthouse
- `2核 2GB 内存`
- `40GB 系统盘`
- `Ubuntu 22.04 LTS`

### 2. 服务器初始化

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-venv python3-pip nginx git
sudo mkdir -p /opt/questionbot/app /opt/questionbot/data/uploads /opt/questionbot/logs
sudo chown -R $USER:$USER /opt/questionbot
```

### 3. 上传代码并安装依赖

```bash
cd /opt/questionbot
python3 -m venv /opt/questionbot/venv
source /opt/questionbot/venv/bin/activate
cd /opt/questionbot/app
pip install -r requirements.txt
```

如果你用 Git：

```bash
git clone <your-repo-url> /opt/questionbot/app
```

### 4. 生产环境变量

在 `/opt/questionbot/app/.env` 中配置：

```env
SECRET_KEY=your-secret
DATABASE_URL=sqlite:////opt/questionbot/data/app.db
UPLOAD_DIR=/opt/questionbot/data/uploads
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your-qdrant-key
QDRANT_COLLECTION=course_kb
SILICONFLOW_API_KEY=your-api-key
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
EMBEDDING_MODEL=BAAI/bge-m3
RERANK_ENABLED=false
SITE_TITLE=HKU Knowledge Companion
SITE_SUBTITLE=Course Design Demonstration
```

### 5. Gunicorn systemd 服务

创建 `/etc/systemd/system/questionbot.service`：

```ini
[Unit]
Description=HKU Knowledge Companion
After=network.target

[Service]
User=root
WorkingDirectory=/opt/questionbot/app
EnvironmentFile=/opt/questionbot/app/.env
ExecStart=/opt/questionbot/venv/bin/gunicorn wsgi:app --bind 127.0.0.1:8000
Restart=always

[Install]
WantedBy=multi-user.target
```

执行：

```bash
sudo systemctl daemon-reload
sudo systemctl enable questionbot
sudo systemctl start questionbot
sudo systemctl status questionbot
```

### 6. Nginx 反向代理

创建 `/etc/nginx/sites-available/questionbot`：

```nginx
server {
    listen 80;
    server_name _;

    location /static/ {
        alias /opt/questionbot/app/app/static/;
        expires 7d;
    }

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

启用并重载：

```bash
sudo ln -s /etc/nginx/sites-available/questionbot /etc/nginx/sites-enabled/questionbot
sudo nginx -t
sudo systemctl reload nginx
```

### 7. 防火墙放行

在 Lighthouse 控制台放通：

- `22`
- `80`
- `443`

### 8. HTTPS（有域名时）

- 购买域名
- 解析 `A` 记录到 Lighthouse 公网 IP
- 在腾讯云申请免费 SSL 证书
- 将证书路径配置到 Nginx 443 虚拟主机中

## Qdrant Cloud 使用建议

1. 注册 Qdrant Cloud
2. 创建 Free Cluster
3. 记录 Cluster URL、API Key
4. 使用一个 collection 即可，例如 `course_kb`

## 测试建议

- 上传文本型 PDF
- 上传 DOCX
- 提问文档内明确出现的问题
- 删除文档后再次提问
- 重启 Gunicorn 与 Nginx，确认服务恢复

## 注意事项

- 这是课程设计演示站，不适合公开生产流量
- DOC 暂不支持，请先转为 DOCX
- 扫描版 PDF 暂不支持 OCR
- 如从旧版本地数据库升级，建议删除旧的 `instance/app.db` 后重新导入文档，以获得完整新字段
