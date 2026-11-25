#!/usr/bin/env bash
# Build the harness and env images for the managed platform, bake each one's
# dev.rlmesh.describe label inside it, and push them.
#
#   docker/publish.sh REGISTRY/NAMESPACE TAG [--no-push] [NAME...]
#
# NAME picks images from the table below (default all); --no-push may come
# anywhere after TAG and keeps the stamped images local instead.
set -euo pipefail

usage() {
  echo "usage: $0 REGISTRY/NAMESPACE TAG [--no-push] [NAME...]" >&2
  exit 2
}
[ $# -ge 2 ] || usage
prefix=$1 tag=$2
shift 2

# name kind target dockerfile [build args...]
images=(
  "llm-harness model harness:LLMHarness harness.Dockerfile"
  "llm-env-chat env chat_env:AdditionTask env-chat.Dockerfile"
  "llm-env-math500 env serve_v1:TasksetEnv env-v1.Dockerfile"
  "llm-env-tau2 env serve_v1:V1EnvEnv env-v1.Dockerfile --target environment --build-arg ENV_ID=primeintellect/tau2-bench"
  "llm-env-wordle env serve_v0:EnvironmentEnv env-v0.Dockerfile --build-arg ENV_ID=wordle"
)
known=" ${images[*]%% *} "

push=true
names=()
for arg in "$@"; do
  if [ "$arg" = --no-push ]; then
    push=false
  elif [[ "$known" == *" $arg "* ]]; then
    names+=("$arg")
  else
    echo "unknown image '$arg'; known:$known" >&2
    exit 2
  fi
done

cd "$(dirname "$0")/.."
iidfile=$(mktemp)
trap 'rm -f "$iidfile"' EXIT
for row in "${images[@]}"; do
  read -r name kind target dockerfile build_args <<<"$row"
  if [ ${#names[@]} -gt 0 ] && [[ " ${names[*]} " != *" $name "* ]]; then
    continue
  fi
  ref=$prefix/$name:$tag
  # shellcheck disable=SC2086 # build_args is a flag list
  docker build --platform linux/amd64 -f "docker/$dockerfile" --iidfile "$iidfile" $build_args .
  iid=$(cat "$iidfile")
  envelope=$(docker run --rm --platform linux/amd64 --entrypoint sh "$iid" -c \
    "python -m rlmesh._describe --$kind '$target' --out /tmp/d.json >/dev/null && cat /tmp/d.json")
  # _describe records a failed build in the envelope and exits 0; admission rejects such a label.
  error=$(python3 -c 'import json, sys
e = json.load(sys.stdin)
for k in ("env_spec", "model_spec"):
    s = e.get(k)
    if isinstance(s, dict) and "error" in s:
        print(k + ": " + str(s["error"]))' <<<"$envelope")
  if [ -n "$error" ]; then
    echo "$name ($iid): describe failed: $error" >&2
    exit 1
  fi
  if [ "$kind" = env ]; then
    entrypoint="[\"python\", \"-m\", \"rlmesh.serve\", \"--env\", \"$target\"]"
  else
    entrypoint="[\"python\", \"-m\", \"rlmesh.serve\", \"$target\"]"
  fi
  # BuildKit takes no image id in FROM, and looks any other name up in its
  # registry first (an unreachable one stalls), so name the base by its id.
  base=rlmesh-publish/$name:${iid#sha256:}
  docker tag "$iid" "$base"
  # The platform registry refuses docker-format manifests, so push the OCI build itself.
  printf 'FROM %s\nENTRYPOINT %s\n' "$base" "$entrypoint" |
    docker build --platform linux/amd64 \
      --output "type=image,name=$ref,oci-mediatypes=true,push=$push" \
      --provenance=false --sbom=false \
      --label "dev.rlmesh.describe=$envelope" -
  docker image rm "$base" >/dev/null
  if [ $push = true ]; then
    echo "pushed $ref"
  else
    echo "built $ref"
  fi
done
