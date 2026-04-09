from fastapi import Body, FastAPI
from pydantic import BaseModel, ConfigDict
from env.core import CacheEnv

app = FastAPI()
env = CacheEnv()


class ResetBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    task_id: str | None = None


@app.post("/reset")
def reset(body: ResetBody = Body(default_factory=ResetBody)):
    state = env.reset(task_id=body.task_id)
    return {
        "state": state,
        "task_id": state.get("task_id"),
    }
@app.post("/step")
def step(action: dict):
    return env.step(action)

@app.get("/state")
def state():
    return env.get_state()