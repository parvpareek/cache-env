---
title: Cache Env
emoji: 🏢
colorFrom: green
colorTo: pink
sdk: docker
pinned: false
---

# Cache invalidation environment (OpenEnv)

## For judges — what this is

**Problem in one sentence:** Backends cache data to go fast; they must decide **when to invalidate, softly refresh, or leave cache alone** using **noisy clues** (like real monitoring), not the ground truth.

**Why it matters:** Cache invalidation is a daily systems tradeoff: act too often and you burn CPU and churn storage; act too late and users see stale data. This env turns that into a **short episode** an agent can be scored on.

**Our approach:** We simulate several cache **items** per episode. Each item has hidden staleness dynamics (TTL, update rate). The API only exposes **observable** fields (`age`, `access_count`, `last_result` as hit/stale with noise). The agent picks an action **per step** for one key: `invalidate`, `refresh`, or `keep`. Step rewards give **partial credit**; at episode end a **grader** produces a **final score in [0, 1]** from correctness, wasted invalidations, and stability.

**Tasks:** Three difficulties — **easy**, **medium**, **hard** — differ by number of items and how volatile hidden state is, so the same policy can be compared across noise levels.

---

## API (OpenEnv-style HTTP)

| Method | Path | Role |
|--------|------|------|
| POST | `/reset` | New episode; returns `state` and `task_id` |
| POST | `/step` | JSON body `{"type":"keep\|refresh\|invalidate","key":"item_0"}`; returns `state`, `reward`, `done`, optional `final_score` when episode ends |
| GET | `/state` | Current observation |

**Deployed Space (example):** `https://parvpareek-cache-env.hf.space` — ping with:

```bash
curl -s -o /dev/null -w '%{http_code}\n' -X POST \
  -H 'Content-Type: application/json' -d '{}' \
  'https://parvpareek-cache-env.hf.space/reset'
```

Expect `200`.

**Local run:** `pip install -r requirements.txt` then `uvicorn app:app --host 0.0.0.0 --port 7860` (or use the Dockerfile).

---

## Baseline inference (`inference.py`)

- Uses the **OpenAI Python client** with **`API_BASE_URL`**, **`MODEL_NAME`**, and **`HF_TOKEN`** (set as environment variables or in a local `.env` loaded by `inference.py`; never commit tokens).
- Talks to the **Space URL** above (override with `ENV_URL` if needed).
- Prints exactly **`[START]`**, one **`[STEP]`** per env step, and **`[END]`** with `score` and `rewards` as required by the challenge spec.

Run:

```bash
export API_BASE_URL='https://router.huggingface.co/v1'
export MODEL_NAME='<model your account can call>'
export HF_TOKEN='hf_...'
python inference.py
```

---

## Validation (pre-submission)

From the repo root:

```bash
openenv validate
./validate-submission.sh 'https://YOUR-SPACE.hf.space' .
docker build .
```

---

## Repository layout (high level)

| Path | Purpose |
|------|---------|
| `app.py` | FastAPI app: `/reset`, `/step`, `/state` |
| `env/` | Environment logic, tasks, grading, generation |
| `openenv.yaml` | OpenEnv metadata |
| `inference.py` | Baseline agent + structured logs |
| `Dockerfile` | Space / CI image |
| `pyproject.toml`, `uv.lock`, `server/app.py` | `openenv validate` / multi-mode layout |

---

## Scoring (short)

- **Per-step reward:** Shaped table (e.g. invalidate when stale is good; invalidate when fresh is penalized). Values can be negative in the middle of an episode.
- **Episode `final_score` (when `done`):** Normalized grader in **[0, 1]** combining decision quality, unnecessary invalidations, and oscillation.

---

## Summary

| Criterion | Status |
|-----------|--------|
| Real-world task (not a toy game) | Cache invalidation under uncertainty |
| `reset` / `step` / `state` | Implemented |
| `openenv.yaml` | Present |
| 3 tasks + grader | `easy` / `medium` / `hard` |
| Meaningful rewards | Dense step reward + episode score in [0, 1] |
| Baseline | `inference.py` + OpenAI client + stdout format |

If anything fails in automated checks, compare your **Space app URL** (`*.hf.space`) and **pushed commit** to what you submit.
