import os
import uuid
import threading
import queue
import torch
import gc
from typing import Optional
from pathlib import Path
from datetime import datetime, timedelta

from dotenv import load_dotenv
from diffusers import ZImagePipeline
from modelscope import snapshot_download

load_dotenv()

MODEL_NAME = "Tongyi-MAI/Z-Image-Turbo"
MODEL_DIR = os.getenv("MODEL_DIR", "models/Z-Image-Turbo")
QUEUE_SIZE = int(os.getenv("QUEUE_SIZE", "10"))
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


class Task:
    def __init__(self, task_id: str, prompt: str, height: int, width: int,
                 num_inference_steps: int, seed: Optional[int]):
        self.task_id = task_id
        self.prompt = prompt
        self.height = height
        self.width = width
        self.num_inference_steps = num_inference_steps
        self.seed = seed
        self.status = "pending"
        self.progress = 0
        self.image_path = None
        self.error = None
        self.created_at = datetime.now()


class ImageGenerationService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.pipe: Optional[ZImagePipeline] = None
        self.task_queue: queue.Queue = queue.Queue(maxsize=QUEUE_SIZE)
        self.tasks: dict[str, Task] = {}
        self.worker_thread: Optional[threading.Thread] = None
        self.running = False
        self.last_request_time = datetime.now()
        self.model_loaded = False

    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self):
        if self.pipe is not None:
            return

        model_path = Path(MODEL_DIR)
        if not model_path.exists() or not any(model_path.iterdir()):
            print(f"Downloading model from ModelScope: {MODEL_NAME}...")
            snapshot_download(MODEL_NAME, local_dir=str(model_path))
            print(f"Model downloaded to: {model_path}")

        print(f"Loading model from: {model_path}")
        dtype = torch.bfloat16 if self._get_device() == "cuda" else torch.float16
        self.pipe = ZImagePipeline.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            local_files_only=True,
        )
        self.pipe.to(self._get_device())
        self.model_loaded = True
        print("Model loaded successfully")

    def _unload_model(self):
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.model_loaded = False
            print("Model unloaded")

    def _worker(self):
        while self.running:
            try:
                task = self.task_queue.get(timeout=1)
            except queue.Empty:
                continue

            try:
                task.status = "running"
                self._generate_image(task)
                task.status = "completed"
            except Exception as e:
                task.status = "failed"
                task.error = str(e)
            finally:
                self.task_queue.task_done()

    def _generate_image(self, task: Task):
        device = self._get_device()
        generator = None
        if task.seed is not None:
            generator = torch.Generator(device).manual_seed(task.seed)

        image = self.pipe(
            prompt=task.prompt,
            height=task.height,
            width=task.width,
            num_inference_steps=task.num_inference_steps,
            guidance_scale=0.0,
            generator=generator,
        ).images[0]

        image_path = OUTPUT_DIR / f"{task.task_id}.png"
        image.save(image_path)
        task.image_path = str(image_path)

    def submit_task(self, prompt: str, height: int, width: int,
                    num_inference_steps: int, seed: Optional[int]) -> tuple[str, bool]:
        self.last_request_time = datetime.now()

        if self.task_queue.full():
            return "", False

        task_id = str(uuid.uuid4())
        task = Task(task_id, prompt, height, width, num_inference_steps, seed)
        self.tasks[task_id] = task

        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker, daemon=True)
            self.worker_thread.start()

        self._load_model()
        self.task_queue.put(task)
        return task_id, True

    def get_status(self, task_id: str) -> Optional[Task]:
        return self.tasks.get(task_id)

    def get_queue_status(self) -> tuple[int, bool]:
        return self.task_queue.qsize(), self.model_loaded

    def unload_model(self):
        self._unload_model()

    def is_idle(self, hours: float = 3) -> bool:
        elapsed = datetime.now() - self.last_request_time
        return elapsed > timedelta(hours=hours)


service = ImageGenerationService()
