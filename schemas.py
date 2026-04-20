from pydantic import BaseModel, Field
from typing import Optional


class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="Image generation prompt")
    height: int = Field(1024, ge=256, le=2048, description="Image height")
    width: int = Field(1024, ge=256, le=2048, description="Image width")
    num_inference_steps: int = Field(9, ge=1, le=50, description="Number of inference steps")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class GenerationResponse(BaseModel):
    task_id: str
    status: str
    message: Optional[str] = None


class TaskStatus(BaseModel):
    task_id: str
    status: str
    progress: Optional[int] = None
    image_path: Optional[str] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    queue_size: int
