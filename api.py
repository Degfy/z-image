from pathlib import Path
import os
import sys
import subprocess
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from schemas import GenerationRequest, GenerationResponse, TaskStatus, HealthResponse
from service import service

BASE_DIR = Path(__file__).parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    service.running = False


app = FastAPI(lifespan=lifespan)


@app.post("/generate", response_model=GenerationResponse)
def generate(req: GenerationRequest):
    task_id, success = service.submit_task(
        prompt=req.prompt,
        height=req.height,
        width=req.width,
        num_inference_steps=req.num_inference_steps,
        seed=req.seed,
    )

    if not success:
        raise HTTPException(status_code=503, detail="Queue is full, try again later")

    return GenerationResponse(task_id=task_id, status="queued", message="Task submitted successfully")


@app.get("/status/{task_id}", response_model=TaskStatus)
def get_status(task_id: str):
    task = service.get_status(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    return TaskStatus(
        task_id=task.task_id,
        status=task.status,
        progress=task.progress,
        image_path=task.image_path,
        error=task.error,
    )


@app.get("/image/{task_id}")
def get_image(task_id: str):
    task = service.get_status(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status != "completed":
        raise HTTPException(status_code=400, detail="Task not completed yet")
    if task.image_path is None:
        raise HTTPException(status_code=500, detail="Image path not found")

    return FileResponse(task.image_path)


@app.post("/unload")
def unload():
    qsize, model_loaded = service.get_queue_status()
    if qsize > 0:
        raise HTTPException(status_code=409, detail=f"Queue has {qsize} pending tasks")
    if model_loaded and service.running:
        tasks_running = any(t.status == "running" for t in service.tasks.values())
        if tasks_running:
            raise HTTPException(status_code=409, detail="Worker is busy, cannot unload model")
    unloaded = service.unload_model()
    if unloaded is None:
        unloaded = True
    if unloaded:
        def _restart():
            import time
            time.sleep(0.5)
            os.execv(sys.executable, [sys.executable] + sys.argv)
        threading.Thread(target=_restart, daemon=True).start()
    return {"unloaded": unloaded}


@app.get("/health", response_model=HealthResponse)
def health():
    qsize, model_loaded = service.get_queue_status()
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        queue_size=qsize,
    )


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "web_ui.html")
