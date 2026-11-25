# LLM agent on Prime Environments Hub tasks

Run [Prime Intellect Environments Hub](https://app.primeintellect.ai/dashboard/environments) tasks on RLMesh with a local LLM served by Ollama. The RLMesh model is an LLM harness; the RLMesh env is a Prime task. One step is one env interaction: a final answer, a game guess, or a tool call. Within a step the harness may call the LLM several times (an empty reply is retried), and every call is logged.

- `harness.py`: `LLMHarness`, the model. Keeps the chat per episode, calls an OpenAI-compatible server, returns the assistant message as the action.
- `chat_env.py`: `ChatTaskEnv`, the env base both servers share, and two tasks that need no verifiers: `AdditionTask`, one question (`python chat_env.py` serves it), and `TicTacToeTask`, two seats played by one model (`--task tictactoe`).
- `serve_v1.py`: serves a verifiers v1 taskset (`--taskset`: `math500`, `humaneval`, `aime25`, `gpqa`), one task per episode scored by the task's own `score`, or a v1 environment (`--env`: `tau2-bench`) that runs its own harness and scoring.
- `serve_v0.py`: serves a verifiers v0 environment (`wordle`, `mastermind`, `alphabet-sort`, `tool-test`, `color-codeword`); the environment runs its own rollout loop and rubric.
- `eval.py`: runs the harness against a served task and prints each episode.
- `fake_llm.py`: a scripted LLM for tests.
- `docker/`, `run_docker.py`: the harness and env servers as images, run as RLMesh sandboxes (see [Docker](#docker)).

Observations and actions are `Text` spaces carrying JSON. An observation is `{"messages": [...], "tools": [...]}`: the full prompt on reset, then only the env's new messages (game feedback, tool results), plus OpenAI tool schemas. An action is one OpenAI-style assistant message, with `tool_calls` when the model calls tools. The final step's `info` holds `is_success`, `reward`, `rewards` (each reward function's score) and `task_id`.

## Setup

Pull a model and start Ollama:

```bash
ollama pull qwen3:4b          # text envs
ollama pull gemma3:4b         # color-codeword (images)
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
    --index-strategy unsafe-first-match \
    $([ $v = v0 ] && echo --override overrides-v0.txt) -r requirements-$v.txt
done
```

`overrides-v0.txt` keeps `verifiers` at 0.1.14 for `color-codeword`, which asks for a dev release it does not need.

From a repository checkout, install this tree's `rlmesh` into each env venv instead (`uv pip install --python .venv-v1/bin/python -e ../../../python/rlmesh`), and run the harness with `uv run --no-sync --with openai python eval.py ...` so the repository environment stays as locked.

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

```bash
.venv-v0/bin/python serve_v0.py --env color-codeword --address 127.0.0.1:5555
python eval.py 127.0.0.1:5555 -n 2 --model gemma3:4b
```

```bash
python chat_env.py --task tictactoe --address 127.0.0.1:5555
python eval.py 127.0.0.1:5555 -n 2 --model qwen3:4b
```

What each task exercises:

- `math500`, `aime25`, `gpqa`: one step. The first reply is the answer.
- `humaneval`: one step, scored by running the tests. Its `passed` reward runs code, so serve it with `--runtime subprocess` (needs `uv` on `PATH` and PyPI access); without it, that reward is skipped. `--runtime subprocess` runs model-written code as the server's user with no sandbox; use it only locally.
- `wordle`, `mastermind`: several steps, one guess each, with game feedback as the next observation. The game ends with a win or when its guesses run out.
- `alphabet-sort`: several steps of instructions and replies.
- `color-codeword` (multimodal): three steps, each observation a user message whose `content` holds two color squares as `image_url` data URLs and a text part; the model decodes and accumulates a codeword. It needs a vision model (`gemma3:4b`; qwen3 reads text only). Rewards are `exact_match_reward` and the `partial_match_score` metric. It always runs to its turn limit, so episodes end truncated. The transcript writes each data URL as `data:<mime>;base64,<N bytes>`; the LLM gets the full image.
- `tictactoe` (multi-agent): seats `X` and `O`, both played by the one harness in one chat. Each observation shows the board and names the seat to move; an invalid or taken cell loses for that seat. `rewards` holds each seat's score (win 1, loss -1, draw 0), `reward` and `is_success` are X's, and `info["winner"]` is `X`, `O` or `draw`. Seeds alternate the first mover.
- `tool-test`: the observation carries tool schemas and the action carries tool calls. It scores the tool calls of the first reply (`max_turns=1`), so its tools never run; `--max-turns 3` lets the env run them and return the results as the next observation, but the reward then scores the final reply.

### Simulated user (tau2-bench)

`tau2-bench` proves an agent that uses tools while talking to an LLM-simulated customer: the env runs tau2's orchestrator, which plays the customer with its own LLM, executes the agent's tool calls against the domain's database (`telecom` by default; `--domain airline`, `retail` or `telecom-workflow` pick another), and scores the final database state against the expected one. Each step is one agent reply; the next observation holds the tool results and the customer's next message, and the tools are the domain's.

The user simulator always asks for `gpt-4.1` at `OPENAI_BASE_URL` with `OPENAI_API_KEY`, which the env process reads; the managed platform injects the same variables. With a Prime API key (`PRIME_API_KEY` or a saved `prime login` in `~/.prime/config.json`) it goes to Prime's billed inference instead and ignores `OPENAI_BASE_URL`. Locally, make Ollama answer to that name:

```bash
ollama cp qwen3:4b gpt-4.1
OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama \
  .venv-v1/bin/python serve_v1.py --env primeintellect/tau2-bench --max-steps 20 --address 127.0.0.1:5555
python eval.py 127.0.0.1:5555 -n 2 --model qwen3:4b --max-tokens 4096
```

The simulator must support tools (`llama3` does not; the telecom customer calls some). A 4B customer often writes `###STOP` without the closing `###`, which tau2 does not count as hanging up, so `--max-steps` ends the chat as truncated after that many agent replies (`stop_condition` `max_steps`, reward 0) instead of running to tau2's own 500-step limit (`tau2_max_steps`, also truncated). A chat the customer ends within the cap is scored normally.

The simulator's calls go from the env straight to that server, so the transcript holds only the agent's side; the simulator's messages appear there as observations. tau2 fetches its domain data with `git` into `~/.cache/tau2-bench` on first start. The final `info` carries `rewards` (`tau2_reward`), `metrics`, `stop_condition` (`user_completed` when the customer ends the chat) and any `errors`. A 4B agent rarely solves a task, so expect a reward of 0. One process serves one tau2 env, since the bridge patches the interception server's module-global client: serve it with `RLMESH_NUM_ENVS=1` (a second instance fails on its first reset).

Hitting the env's turn limit (`max_turns`) ends the episode as truncated when that limit is above 1; single-turn envs (`max_turns=1`) end there normally.

qwen3 thinks before it answers and that counts against `--max-tokens` (default 8192). Ollama returns the reasoning separately; the transcript keeps it and the action leaves it out. RLMesh abandons a step after 300 s, so keep `--max-tokens` to what your server generates in that time; the harness skips a retry that wouldn't fit.

## Output

Both sides write under `$RLMESH_OUTPUT_DIR` (default `./runs/<timestamp>`):

- `transcripts/<episode_id>.jsonl` (harness): one line per LLM call, `{"step", "call", "request": {"model", "messages", "tools"}, "response", "usage", "latency_ms", "ts"}` (a failed call has `"error"` in place of `"response"` and `"usage"`), then `{"event": "episode_end", "steps", "calls", ...}`.
- `env/seed-<seed>.jsonl` (env server): one line per finished episode with its rewards. The env doesn't see the episode id, so these are keyed by seed.

## Docker

The harness and each env server also ship as images, run the way RLMesh Managed runs them, with `python -m rlmesh.serve`: `harness.Dockerfile` serves `LLMHarness`, `env-chat.Dockerfile` `AdditionTask`, `env-v1.Dockerfile` a taskset (`serve_v1:TasksetEnv`; its `environment` target serves `serve_v1:V1EnvEnv`), and `env-v0.Dockerfile` an environment (`serve_v0:EnvironmentEnv`). Each env class takes its settings from the variables below, so it builds with no arguments, and loads its taskset or environment on the first reset. Build them from this directory:

```bash
docker build -f docker/harness.Dockerfile -t llm-harness .
docker build -f docker/env-v0.Dockerfile -t llm-env-v0 .
docker build -f docker/env-v1.Dockerfile -t llm-env-v1 .
docker build -f docker/env-v1.Dockerfile -t llm-env-humaneval \
  --build-arg TASKSET=primeintellect/humaneval --build-arg RUNTIME=subprocess .
docker build -f docker/env-v1.Dockerfile -t llm-env-tau2 --target environment \
  --build-arg ENV_ID=primeintellect/tau2-bench .
```

The tau2 image's user simulator needs `OPENAI_BASE_URL` and `OPENAI_API_KEY` at run time (`docker run -e`), which a sandbox doesn't pass through.

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
| `ENV_ID`            | env-v1 (build arg)  | empty; the `environment` target's environment (`primeintellect/tau2-bench`)     |
| `DOMAIN`            | env-v1 (run time)   | empty; `--domain` for the `environment` target (`docker run -e`)                |
| `MAX_STEPS`         | env-v1 (build arg)  | empty; `--max-steps` for the `environment` target                               |
| `MAX_TURNS`         | env-v0 (build arg)  | empty; `--max-turns`                                                            |
| `RLMESH_OUTPUT_DIR` | all                 | `/rlmesh/output/model` (harness), `/rlmesh/output/env` (env)                    |

Each image sets `RLMESH_READY_FILE`, so `docker ps` shows it healthy once it serves.

### Managed platform

`docker/publish.sh` builds each image for `linux/amd64` (the platform runs only that), bakes its `dev.rlmesh.describe` label inside the image, where the class builds, and pushes it as an OCI image, which the platform registry requires:

```bash
NS=registry.example.com/myteam
docker/publish.sh "$NS" v1             # all five images
docker/publish.sh "$NS" v1 --no-push   # build and stamp only
rlmesh check-image "$NS/llm-env-chat:v1"
```

It publishes `llm-harness` (model), `llm-env-chat`, `llm-env-math500`, `llm-env-tau2` and `llm-env-wordle`; names after the tag (and `--no-push`) publish only those. No key is baked: the platform injects `OPENAI_*` and `ANTHROPIC_*` into the model container from an `api_key` connection named in `models[].connections`. The push-time probe runs with no key and sends random text, which the harness answers with an empty message and the chat env scores 0. An evaluation request:

```json
{
  "models": [{ "model": "<ns>/llm-harness", "connections": ["conn_…"] }],
  "tasks": [{ "taskId": "math500/0", "env": "<ns>/llm-env-math500", "episodes": 2 }]
}
```
