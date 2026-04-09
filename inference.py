import json
import os
import sys
import textwrap
from pathlib import Path
from typing import List, Optional

import requests
from openai import OpenAI

from env.grader import clamp_strict_unit_interval

# Load .env from repo root so HF_TOKEN / API_BASE_URL work when you run: python inference.py
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

# ---- Mandatory env (see hackathon spec) ----
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
# HF deprecated api-inference.huggingface.co (410); router is the supported OpenAI-compatible host.

MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

ENV_URL = os.getenv("ENV_URL", "https://parvpareek-cache-env.hf.space")
BENCHMARK = "cache_invalidation_env"

if not API_KEY:
    print(
        "WARNING: HF_TOKEN is not set. LLM calls will fail; the script will fall back to the "
        "heuristic policy. Set HF_TOKEN in the environment or in a .env file next to inference.py.",
        file=sys.stderr,
    )

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "hf-invalid")

MEMORY = {}
LAST_USED = None

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a cache invalidation agent. Given the environment state (JSON), reply with exactly one JSON object
    on a single line, no markdown, with keys "type" and "key". type must be one of: invalidate, refresh, keep.
    key must match one of the item keys in state["items"].
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def select_item(state, step):
    global LAST_USED
    items = state["items"]

    def score(item):
        s = 0
        if item["last_result"] == "stale":
            s += 3
        if item["age"] > 5:
            s += 2
        if item["access_count"] > 10:
            s += 1
        return s

    best = max(items, key=score)

    if step % 2 == 1:
        for item in items:
            if item["key"] != LAST_USED:
                LAST_USED = item["key"]
                return item

    LAST_USED = best["key"]
    return best


def decide(item, step):
    key = item["key"]
    last_result = item["last_result"]
    age = item["age"]

    mem = MEMORY.get(key, {})

    if mem.get("last_action") == "invalidate" and step - mem.get("last_step", -10) < 2:
        return {"type": "keep", "key": key}

    if last_result == "stale" and age > 2:
        return {"type": "invalidate", "key": key}

    if 3 <= age <= 6:
        return {"type": "refresh", "key": key}

    if last_result == "hit" and age < 3:
        return {"type": "keep", "key": key}

    if age > 6:
        return {"type": "refresh", "key": key}

    return {"type": "keep", "key": key}


def llm_action(state) -> Optional[dict]:
    """Call HF OpenAI-compatible API; return None on any failure so caller can fall back."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"State:\n{json.dumps(state)}\n\n"
                        'Return JSON only: {"type": "...", "key": "..."}'
                    ),
                },
            ],
            temperature=0,
            max_tokens=150,
        )
        text = (completion.choices[0].message.content or "").strip()
        if text.startswith("```"):
            parts = text.split("```")
            text = parts[1] if len(parts) >= 2 else text
            text = text.strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
        action = json.loads(text)
        if "type" in action and "key" in action:
            return {"type": action["type"], "key": action["key"]}
    except Exception as exc:
        print(f"[LLM] request/parse failed: {exc}", file=sys.stderr)
    return None


def run() -> None:
    global LAST_USED
    LAST_USED = None
    MEMORY.clear()

    rewards: List[float] = []
    steps_taken = 0
    episode_score = 0.0
    success = False

    try:
        score_from_env = False
        res = requests.post(
            f"{ENV_URL}/reset",
            json={},
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        res.raise_for_status()
        body = res.json()
        state = body.get("state", body)
        task_id = str(body.get("task_id", "unknown"))

        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

        for step in range(1, 11):
            item = select_item(state, step)

            action = llm_action(state)
            if action is None:
                action = decide(item, step)

            MEMORY[item["key"]] = {
                "last_action": action["type"],
                "last_step": step,
            }

            step_res = requests.post(
                f"{ENV_URL}/step",
                json=action,
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            step_res.raise_for_status()
            data = step_res.json()

            reward = float(data["reward"])
            done = bool(data["done"])
            rewards.append(reward)
            steps_taken = step

            if data.get("final_score") is not None:
                episode_score = float(data["final_score"])
                score_from_env = True

            log_step(
                step=step,
                action=json.dumps(action),
                reward=reward,
                done=done,
                error=None,
            )

            state = data["state"]

            if done:
                break

        if rewards:
            avg_r = sum(rewards) / len(rewards)
            success = avg_r > 0.3
        if not score_from_env and rewards:
            avg_r = sum(rewards) / len(rewards)
            episode_score = max(0.0, min(1.0, (avg_r + 1.0) / 2.0))

    except Exception as exc:
        success = False
        print(f"[RUN] fatal: {exc}", file=sys.stderr)
    finally:
        episode_score = clamp_strict_unit_interval(episode_score)
        log_end(
            success=success,
            steps=steps_taken,
            score=episode_score,
            rewards=rewards,
        )


if __name__ == "__main__":
    run()
