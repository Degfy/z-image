# Z-Image-Turbo REST Service

基于 [Z-Image-Turbo](https://github.com/Tongyi-MAI/Z-Image) 的图像生成 REST 服务。

## 配置

在 `.env` 文件中配置：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `MODEL_NAME` | 模型名称 | `Tongyi-MAI/Z-Image-Turbo` |
| `MODEL_SOURCE` | 模型来源：`modelscope` 或 `huggingface` | `modelscope` |
| `MODEL_CACHE_DIR` | 模型缓存目录，默认为空使用系统路径 | 空 |
| `HOST` | 服务监听地址 | `0.0.0.0` |
| `PORT` | 服务监听端口 | `8000` |
| `QUEUE_SIZE` | 任务队列长度 | `10` |

## 运行

```bash
uv sync
uv run python main.py
```

首次调用 API 时模型会自动从 modelscope 下载。

## 接口

- `POST /generate` - 提交图像生成任务
  - `prompt`: 生成提示词
  - `height`: 图像高度 (256-2048)
  - `width`: 图像宽度 (256-2048)
  - `num_inference_steps`: 推理步数 (1-50)
  - `seed`: 随机种子（可选）

- `GET /status/{task_id}` - 查询任务状态

- `GET /image/{task_id}` - 获取生成的图片

- `POST /unload` - 卸载模型并重启服务

- `GET /health` - 健康检查

## 示例

```bash
# 提交任务
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a cute cat","height":1024,"width":1024}'

# 查询状态
curl http://localhost:8000/status/{task_id}

# 获取图片
curl http://localhost:8000/image/{task_id} -o output.png
```

## 硬件支持

- NVIDIA GPU (CUDA)
- Apple Silicon (MPS)
- CPU
