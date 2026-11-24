# LLM agent on Prime Environments Hub tasks

Run [Prime Intellect Environments Hub](https://app.primeintellect.ai/dashboard/environments) tasks on RLMesh with a local LLM served by Ollama. The RLMesh model is an LLM harness; the RLMesh env is a Prime task. One step is one env interaction: a final answer, a game guess, or a tool call. Within a step the harness may call the LLM several times (an empty reply is retried), and every call is logged.

- `harness.py`: `LLMHarness`, the model. Keeps the chat per episode, calls an OpenAI-compatible server, returns the assistant message as the action.
- `chat_env.py`: `ChatTaskEnv`, the env base both servers share, and `AdditionTask`, a one-question task that needs no verifiers (`python chat_env.py` serves it).
- `serve_v1.py`: serves a verifiers v1 taskset (`math500`, `humaneval`, `aime25`, `gpqa`); one task per episode, scored by the task's own `score`.
- `serve_v0.py`: serves a verifiers v0 environment (`wordle`, `mastermind`, `alphabet-sort`, `tool-test`); the environment runs its own rollout loop and rubric.
- `eval.py`: runs the harness against a served task and prints each episode.
- `fake_llm.py`: a scripted LLM for tests.
- `docker/`, `run_docker.py`: the harness and env servers as images, run as RLMesh sandboxes (see [Docker](#docker)).

Observations and actions are `Text` spaces carrying JSON. An observation is `{"messages": [...], "tools": [...]}`: the full prompt on reset, then only the env's new messages (game feedback, tool results), plus OpenAI tool schemas. An action is one OpenAI-style assistant message, with `tool_calls` when the model calls tools. The final step's `info` holds `is_success`, `reward`, `rewards` (each reward function's score) and `task_id`.

## Setup

Pull a model and start Ollama:

```bash
ollama pull qwen3:4b
ollama serve
```

The harness reads `OPENAI_BASE_URL` (default `http://localhost:11434/v1`) and `OPENAI_API_KEY` (default `ollama`), so any OpenAI-compatible server works. It needs `rlmesh` and `openai`:

```bash
uv pip install --pre "rlmesh>=0.2.0a1" openai
```

verifiers v1 and v0 can't share an interpreter, so each env server gets its own venv; RLMesh's process boundary keeps them apart from the harness. Hub packages come from Prime's index (`--extra-index-url` would resolve `wordle` to an unrelated PyPI package):

```bash
for v in v1 v0; do
  uv venv --python 3.11 .venv-$v
  uv pip install --python .venv-$v/bin/python \
    --index https://hub.primeintellect.ai/primeintellect/simple \
    --index-strategy unsafe-first-match -r requirements-$v.txt
done
```

From a repository checkout, install this tree's `rlmesh` into each env venv instead (`uv pip install --python .venv-v1/bin/python -e ../../../python/rlmesh`), and run the harness with `uv run --with openai python eval.py ...` so the repository environment stays as locked.

## Run

Serve a task in one terminal and evaluate it in another, from this directory. Seeds `0..n-1` pick the tasks, so a run reproduces; `eval.py` always seeds, and an unseeded reset gets task 0.

```bash
.venv-v1/bin/python serve_v1.py --taskset primeintellect/math500 --address 127.0.0.1:5555
python eval.py 127.0.0.1:5555 -n 3 --model qwen3:4b
```

```bash
.venv-v0/bin/python serve_v0.py --env wordle --address 127.0.0.1:5555
python eval.py 127.0.0.1:5555 -n 2
```

What each task exercises:

- `math500`, `aime25`, `gpqa`: one step. The first reply is the answer.
- `humaneval`: one step, scored by running the tests. Its `passed` reward runs code, so serve it with `--runtime subprocess` (needs `uv` on `PATH` and PyPI access); without it, that reward is skipped. `--runtime subprocess` runs model-written code as the server's user with no sandbox; use it only locally.
- `wordle`, `mastermind`: several steps, one guess each, with game feedback as the next observation. The game ends with a win or when its guesses run out.
- `alphabet-sort`: several steps of instructions and replies.
- `tool-test`: the observation carries tool schemas and the action carries tool calls. It scores the tool calls of the first reply (`max_turns=1`), so its tools never run; `--max-turns 3` lets the env run them and return the results as the next observation, but the reward then scores the final reply.

Hitting the env's turn limit (`max_turns`) ends the episode as truncated when that limit is above 1; single-turn envs (`max_turns=1`) end there normally.

qwen3 thinks before it answers and that counts against `--max-tokens` (default 8192). Ollama returns the reasoning separately; the transcript keeps it and the action leaves it out. RLMesh abandons a step after 300 s, so keep `--max-tokens` to what your server generates in that time; the harness skips a retry that wouldn't fit.

## Output

Both sides write under `$RLMESH_OUTPUT_DIR` (default `./runs/<timestamp>`):

- `transcripts/<episode_id>.jsonl` (harness): one line per LLM call, `{"step", "call", "request": {"model", "messages", "tools"}, "response", "usage", "latency_ms", "ts"}` (a failed call has `"error"` in place of `"response"` and `"usage"`), then `{"event": "episode_end", "steps", "calls", ...}`.
- `env/seed-<seed>.jsonl` (env server): one line per finished episode with its rewards. The env doesn't see the episode id, so these are keyed by seed.

## Docker

The harness and each env server also ship as images, run the way RLMesh Managed runs them: `harness.Dockerfile` serves `LLMHarness` with `python -m rlmesh.serve`, and `env-v1.Dockerfile` and `env-v0.Dockerfile` serve a taskset and an environment. Build them from this directory:

```bash
docker build -f docker/harness.Dockerfile -t llm-harness .
docker build -f docker/env-v0.Dockerfile -t llm-env-v0 .
docker build -f docker/env-v1.Dockerfile -t llm-env-v1 .
docker build -f docker/env-v1.Dockerfile -t llm-env-humaneval \
  --build-arg TASKSET=primeintellect/humaneval --build-arg RUNTIME=subprocess .
```

`RUNTIME=subprocess` runs model-written code inside the env container as the server's user, so that code can read the container's credentials (`RLMESH_HOST_IDENTITY`); use it only locally.

The harness image reaches Ollama at `host.docker.internal`, which only Docker Desktop provides. On Linux, build it with `--build-arg OPENAI_BASE_URL=http://172.17.0.1:11434/v1` (the Docker bridge gateway; `docker network inspect bridge` shows yours) and start Ollama with `OLLAMA_HOST=0.0.0.0`, since it binds `127.0.0.1` by default.

`run_docker.py` starts the env image and the harness image as sandboxes and runs one against the other; `--build v0` or `--build v1` builds both first (the env tag defaults to `llm-env-<v0|v1>`). Each container mounts only its own part of one host directory, the layout of [Output](#output): the harness `transcripts/`, the env `env/`. It needs a POSIX host (`os.getuid()`); on Windows, use WSL.

```bash
python run_docker.py --env-image llm-env-v0 -n 1
python run_docker.py --env-image llm-env-humaneval -n 1 --output runs/humaneval
```

The images install `rlmesh==0.2.0a1` from PyPI, and the host must run the same build: a repository checkout's dev build speaks only to itself, so run `run_docker.py` from a venv with `rlmesh==0.2.0a1` installed (`--build-arg RLMESH_VERSION=...` pins another release).

A sandbox env joined by a sandbox model is reached on the Docker bridge network, which Docker Desktop (macOS, Windows) doesn't route to from the host, so there the run times out waiting for the session host. Either run the harness on the host against the env image (`--local-harness`, needs `openai`), or run `run_docker.py` itself in a Linux container on the Docker VM's network. That needs Docker Desktop 4.34 or later with host networking enabled (Settings → Resources → Network), and the mounted Docker socket gives the driver control of the daemon:

```bash
docker build -t llm-driver - <<'EOF'
FROM python:3.11-slim
COPY --from=docker:28.0.4-cli /usr/local/bin/docker /usr/local/bin/docker
RUN pip install --no-cache-dir rlmesh==0.2.0a1
EOF
docker run --rm --network host -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$PWD:$PWD" -w "$PWD" llm-driver python run_docker.py --env-image llm-env-v0
```

A sandbox passes no environment variables through, so the images bake these in (`docker run -e` overrides them when you run a container yourself, as in [byo_container](../byo_container/README.md#running-the-container-yourself)). Never bake a real `OPENAI_API_KEY` into an image: the sandbox path is for local or keyless servers, and the managed platform injects credentials at run time.

| Variable            | Image               | Default                                                                         |
| ------------------- | ------------------- | ------------------------------------------------------------------------------- |
| `OPENAI_BASE_URL`   | harness (build arg) | `http://host.docker.internal:11434/v1` (Ollama on the Docker host)              |
| `OPENAI_API_KEY`    | harness             | `ollama`                                                                        |
| `OPENAI_MODEL`      | harness (build arg) | `qwen3:4b`                                                                      |
| `OPENAI_MAX_TOKENS` | harness             | `8192`                                                                          |
| `TASKSET`           | env-v1 (build arg)  | `primeintellect/math500`                                                        |
| `RUNTIME`           | env-v1 (build arg)  | empty; `subprocess` runs humaneval's tests in the container (needs PyPI access) |
| `ENV_ID`            | env-v0 (build arg)  | `wordle`                                                                        |
| `RLMESH_OUTPUT_DIR` | all                 | `/rlmesh/output/model` (harness), `/rlmesh/output/env` (env)                    |

Each image sets `RLMESH_READY_FILE`, so `docker ps` shows it healthy once it serves.
