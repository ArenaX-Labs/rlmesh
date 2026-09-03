# Submit Evaluations to a Platform

A managed RLMesh platform runs evaluations for you: it schedules model servers and environment workers on a cluster, records every episode, and keeps the results. This page covers driving one from the CLI and from Python. Both use the session that `rlmesh login` stores, so nothing here takes an API key.

```
rlmesh login                      # once; opens the browser device flow
rlmesh whoami                     # profile, platform, signed-in state
```

## The request

An evaluation request is a JSON document: the models to run, the tasks to run them on, and the execution knobs. The platform's OpenAPI contract (`EvaluationRequest`) is the full field list; this is a representative one:

```json
{
  "models": [{ "model": "prebuilt/xvla/libero" }],
  "tasks": [
    { "env": "prebuilt/libero-mujoco330", "taskId": "libero_90/0", "episodes": 10, "seed": 0 }
  ],
  "maxBatchSize": 32,
  "clientsPerServer": 60,
  "timeoutSeconds": 21600,
  "tags": { "experiment": "smoke" },
  "metadata": { "evaluationLabel": "xvla smoke" }
}
```

Tags are the handle you list and filter by later; `metadata` is free text the dashboard and reports display.

## CLI

```
rlmesh eval submit request.json --preview   # validate and size, launch nothing
rlmesh eval submit request.json             # launch; prints the id and dashboard link
rlmesh eval submit request.json --wait      # launch and block until it finishes
rlmesh eval list --tag experiment:smoke     # newest first; --status, --q, --limit, --json
rlmesh eval get eval_...                    # the evaluation as JSON
rlmesh eval wait eval_...                   # poll; exits nonzero unless completed
rlmesh eval cancel eval_...
```

`rlmesh eval submit -` reads the request from stdin, so a generator script can pipe straight in. Every command takes `--profile` to target another platform.

Scripts that call the HTTP API themselves can borrow the session instead of managing a key:

```
curl -H "Authorization: Bearer $(rlmesh token)" https://api.rlmesh.dev/v1/evaluations
```

## Python

{mod}`rlmesh.platform` wraps the same routes with no dependencies beyond the standard library. {class}`~rlmesh.platform.Client` finds the CLI's session on its own and refreshes it when the token expires.

```python
from rlmesh import platform

client = platform.Client()

request = {
    "models": [{"model": "prebuilt/xvla/libero"}],
    "tasks": platform.scenario_tasks(
        "prebuilt/libero-mujoco330", "libero_90", tasks=90, episodes=40, workers=4
    ),
    "managed": {"modelReplicas": 6, "maxParallelWorkloads": 360},
    "maxBatchSize": 32,
    "clientsPerServer": 60,
    "tags": {"experiment": "fleet"},
    "metadata": {"evaluationLabel": "xvla fleet 6gpu libero90x40"},
}

print(client.preview(request)["estimates"])
submitted = client.submit(request)
final = client.wait(submitted["id"])
print(final["status"], final["progress"])

for evaluation in client.evaluations(tags={"experiment": "fleet"}, status="completed"):
    print(evaluation["id"], evaluation["metadata"].get("evaluationLabel"))
```

{func}`~rlmesh.platform.scenario_tasks` expands a scenario group into one task each. Its `workers` argument fills the request's `workers` field, which asks the platform to split a task's episodes across that many environment workers without changing which episodes run. `client.get(path, **query)` and `client.post(path, body)` reach any other `/v1` route.

Set `RLMESH_API_KEY` (and optionally `RLMESH_PLATFORM_URL`) to run without a signed-in CLI, for example in CI.
