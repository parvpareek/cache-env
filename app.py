from fastapi import FastAPI
from env.core import CacheEnv

app = FastAPI()
env = CacheEnv()

@app.post("/reset")
def reset():
    state = env.reset()
    return {
        "state": state,
        "task_id": state.get("task_id")
    }
@app.post("/step")
def step(action: dict):
    return env.step(action)

@app.get("/state")
def state():
    return env.get_state()