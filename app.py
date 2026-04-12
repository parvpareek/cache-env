from fastapi import Body, FastAPI
from pydantic import BaseModel, ConfigDict
from env.core import CacheEnv
from env.tasks import TASK_MANIFEST

app = FastAPI()
env = CacheEnv()


class ResetBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    task_id: str | None = None
    task_name: str | None = None


@app.post("/reset")
def reset(body: ResetBody = Body(default_factory=ResetBody)):
    task_key = body.task_id or body.task_name
    state = env.reset(task_id=task_key)
    return {
        "state": state,
        "task_id": state.get("task_id"),
    }


@app.get("/tasks")
def list_tasks():
    """Hub validators use this to discover tasks that expose episode grading (final_score)."""
    return {"tasks": TASK_MANIFEST}


@app.post("/step")
def step(action: dict):
    return env.step(action)


@app.get("/state")
def state():
    return env.get_state()