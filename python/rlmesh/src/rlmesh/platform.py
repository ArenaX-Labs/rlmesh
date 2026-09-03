"""Submit and read evaluations on a managed RLMesh platform.

Authentication comes from the signed-in ``rlmesh`` CLI (``rlmesh login``), or
from ``RLMESH_API_KEY`` when set. Requests and responses are the platform's own
JSON shapes; ``EvaluationRequest`` in the platform's OpenAPI contract lists
every field a request accepts.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator, Mapping
from typing import Any

__all__ = ["Client", "PlatformError", "scenario_tasks"]

TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


class PlatformError(RuntimeError):
    """An HTTP error from the platform, with the server's message."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


class Client:
    """A thin client over the platform's ``/v1`` HTTP API.

    ``Client()`` uses the CLI's default profile; ``Client(profile="staging")``
    another one. Pass ``url`` and ``token`` to bypass the CLI entirely
    (``RLMESH_API_KEY`` with ``RLMESH_PLATFORM_URL`` does the same from the
    environment).
    """

    def __init__(
        self,
        *,
        profile: str | None = None,
        url: str | None = None,
        token: str | None = None,
    ) -> None:
        self._profile = profile
        self._token = token or os.environ.get("RLMESH_API_KEY")
        self._url = url or os.environ.get("RLMESH_PLATFORM_URL")
        if self._url is None and self._token is None:
            self._refresh_from_cli()
        if self._url is None:
            self._url = "https://api.rlmesh.dev"
        self._url = self._url.rstrip("/")

    @property
    def url(self) -> str:
        """The platform base URL."""
        return self._url or ""

    # -- generic --------------------------------------------------------

    def get(self, path: str, **query: Any) -> Any:
        """``GET /v1/...``; ``query`` values may be lists for repeatable keys."""
        return self._request("GET", path, query=query)

    def post(self, path: str, body: Mapping[str, Any] | None = None) -> Any:
        """``POST /v1/...`` with a JSON body."""
        return self._request("POST", path, body=body)

    def patch(self, path: str, body: Mapping[str, Any]) -> Any:
        """``PATCH /v1/...`` with a JSON body."""
        return self._request("PATCH", path, body=body)

    # -- evaluations ----------------------------------------------------

    def preview(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and size a request without launching it."""
        return self.post("/v1/evaluation-previews", request)

    def submit(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Launch an evaluation; returns ``{"id", "status", "warnings", ...}``."""
        return self.post("/v1/evaluations", request)

    def evaluation(self, eval_id: str) -> dict[str, Any]:
        """One evaluation: status, progress, request, and effective config."""
        return self.get(f"/v1/evaluations/{eval_id}")

    def evaluations(
        self,
        *,
        status: str | None = None,
        tags: Mapping[str, str] | None = None,
        q: str | None = None,
        limit: int = 100,
    ) -> Iterator[dict[str, Any]]:
        """Iterate evaluations newest first, following pagination."""
        query: dict[str, Any] = {"limit": limit}
        if status:
            query["status"] = status
        if q:
            query["q"] = q
        if tags:
            query["tag"] = [f"{k}:{v}" for k, v in tags.items()]
        while True:
            page = self.get("/v1/evaluations", **query)
            yield from page.get("items", [])
            cursor = page.get("nextCursor")
            if not cursor:
                return
            query["cursor"] = cursor

    def wait(
        self,
        eval_id: str,
        *,
        poll_seconds: float = 15.0,
        timeout: float | None = None,
        progress: bool = True,
    ) -> dict[str, Any]:
        """Poll until the evaluation reaches a terminal status."""
        deadline = None if timeout is None else time.monotonic() + timeout
        last = ""
        while True:
            evaluation = self.evaluation(eval_id)
            done = evaluation.get("progress", {})
            line = (
                f"{evaluation.get('status')}  "
                f"{done.get('completedEpisodes')}/{done.get('totalEpisodes')} episodes"
            )
            if progress and line != last:
                print(line, file=sys.stderr, flush=True)
                last = line
            if evaluation.get("status") in TERMINAL_STATUSES:
                return evaluation
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"{eval_id} still {evaluation.get('status')}")
            time.sleep(poll_seconds)

    def cancel(self, eval_id: str) -> dict[str, Any]:
        """Cancel a running evaluation."""
        return self.post(f"/v1/evaluations/{eval_id}/cancellations", {})

    def scores(self, eval_id: str, *, group_by: str = "task") -> dict[str, Any]:
        """Outcome rollups grouped by task, scenario, benchmark, perturbation, or model."""
        return self.get(f"/v1/evaluations/{eval_id}/scores", groupBy=group_by)

    def metrics(self, eval_id: str) -> dict[str, Any]:
        """Throughput and timing rollups over every task."""
        return self.get(f"/v1/evaluations/{eval_id}/metrics")

    # -- transport ------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, Any] | None = None,
        body: Mapping[str, Any] | None = None,
        retry_auth: bool = True,
    ) -> Any:
        url = f"{self._url}{path}"
        if query:
            pairs = [(k, v) for k, v in query.items() if v is not None]
            url += "?" + urllib.parse.urlencode(pairs, doseq=True)
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self._token}")
        req.add_header("Accept", "application/json")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            if error.code == 401 and retry_auth and self._refresh_from_cli():
                return self._request(
                    method, path, query=query, body=body, retry_auth=False
                )
            raise PlatformError(error.code, _error_message(error.read())) from None
        return json.loads(raw) if raw else None

    def _refresh_from_cli(self) -> bool:
        """Pull a fresh token (and the platform URL) from the signed-in CLI."""
        if os.environ.get("RLMESH_API_KEY"):
            return False
        argv = [sys.executable, "-m", "rlmesh", "token", "--json"]
        if self._profile:
            argv += ["--profile", self._profile]
        completed = subprocess.run(argv, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                completed.stderr.strip() or "rlmesh token failed; run `rlmesh login`"
            )
        session = json.loads(completed.stdout)
        self._token = session["token"]
        self._url = self._url or session["platform"]
        return True


def scenario_tasks(
    env: str,
    scenario: str,
    *,
    tasks: int | list[int] | None = None,
    task_ids: list[str] | None = None,
    episodes: int = 10,
    seed: int = 0,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """One ``tasks`` entry per scenario of a group.

    Pass ``tasks=90`` for ``libero_90/0..89``, a list for a subset, or
    ``task_ids`` for explicit scenario names. ``workers`` asks the platform to
    split each task's episodes across that many environment workers; the split
    never changes which episodes run.
    """
    if task_ids is None:
        indexes = range(tasks) if isinstance(tasks, int) else list(tasks or [])
        task_ids = [f"{scenario}/{index}" for index in indexes]
    return [
        {
            "env": env,
            "taskId": task_id,
            "episodes": episodes,
            "seed": seed,
            **({"workers": workers} if workers > 1 else {}),
            "metadata": {"environmentId": f"{env}:{task_id}"},
        }
        for task_id in task_ids
    ]


def _error_message(body: bytes) -> str:
    try:
        return json.loads(body)["error"]["message"]
    except (ValueError, KeyError, TypeError):
        return body.decode(errors="replace")[:300]
