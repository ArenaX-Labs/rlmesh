#!/usr/bin/env bash
set -euo pipefail
# Never enable tracing. Credentials exist only on the ephemeral release runner.
: "${RUNNER_TEMP:?release runner scratch is required}"
keychain="$RUNNER_TEMP/rlmesh-signing.keychain-db"
mode="${1:-sign}"
shift || true
paths=("$@")
[[ ${#paths[@]} -gt 0 ]] || paths=(target/macos-universal/librlmesh_capi.dylib)
evidence="$(dirname "${paths[0]}")"
identifier=()
if [[ -n "${RLMESH_CODESIGN_IDENTIFIER:-}" ]]; then identifier=(--identifier "$RLMESH_CODESIGN_IDENTIFIER"); fi
search_list="$RUNNER_TEMP/rlmesh-keychain-search-list"
if [[ "$mode" == cleanup ]]; then
  if [[ -f "$search_list" ]]; then
    keychains=()
    while IFS= read -r line; do
      if [[ -n "$line" ]]; then keychains+=("$line"); fi
    done < "$search_list"
    security list-keychains -d user -s ${keychains[@]+"${keychains[@]}"} || true
    rm -f "$search_list"
  fi
  security delete-keychain "$keychain" 2>/dev/null || true
  rm -f "$RUNNER_TEMP/rlmesh-signing.p12" "$RUNNER_TEMP/rlmesh-notary.p8" "$RUNNER_TEMP/rlmesh-notary.zip"
  exit 0
fi

if [[ "$mode" == sign ]]; then
  rm -f "$evidence/notarization.json" "$evidence/notarization-log.json"
  missing=false
  for secret in APPLE_DEVELOPER_ID_P12 APPLE_DEVELOPER_ID_P12_PASSWORD APPLE_TEAM_ID; do
    if [[ -z "${!secret:-}" ]]; then missing=true; fi
  done
  notary=0
  for secret in APPLE_NOTARY_KEY APPLE_NOTARY_KEY_ID APPLE_NOTARY_ISSUER_ID; do
    if [[ -n "${!secret:-}" ]]; then notary=$((notary + 1)); fi
  done
  notarize=true
  if [[ "$notary" -eq 0 && "${RLMESH_MACOS_NOTARIZE:-}" == optional ]]; then
    notarize=false
  elif [[ "$notary" -ne 3 ]]; then
    missing=true
  fi
  if [[ "$missing" == true ]]; then
    if [[ "${GITHUB_REF:-}" == refs/tags/v* && "${RLMESH_ALLOW_UNSIGNED_MACOS:-}" != true ]]; then
      echo '::error::Apple signing/notarization secrets are incomplete; tag releases require them unless RLMESH_ALLOW_UNSIGNED_MACOS is explicitly true.' >&2
      exit 1
    fi
    echo '::warning::Apple secrets are incomplete; explicitly allowing ad-hoc signed macOS binaries without notarization.'
    codesign --force --sign - ${identifier[@]+"${identifier[@]}"} "${paths[@]}"
    codesign --verify --strict --verbose=2 "${paths[@]}"
    if [[ -n "${GITHUB_OUTPUT:-}" ]]; then printf 'signed=false\nnotarize=false\n' >> "$GITHUB_OUTPUT"; fi
    exit 0
  fi
  python - <<'PY'
import base64
import os
from pathlib import Path
root = Path(os.environ["RUNNER_TEMP"])
files = [("rlmesh-signing.p12", base64.b64decode("".join(os.environ["APPLE_DEVELOPER_ID_P12"].split()), validate=True))]
if os.environ.get("APPLE_NOTARY_KEY"):
    files.append(("rlmesh-notary.p8", os.environ["APPLE_NOTARY_KEY"].encode()))
for name, contents in files:
    descriptor = os.open(root / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "wb") as file:
        file.write(contents)
PY
  keychain_password="$(openssl rand -hex 32)"
  security create-keychain -p "$keychain_password" "$keychain"
  security set-keychain-settings -lut 21600 "$keychain"
  security unlock-keychain -p "$keychain_password" "$keychain"
  security import "$RUNNER_TEMP/rlmesh-signing.p12" -k "$keychain" \
    -P "$APPLE_DEVELOPER_ID_P12_PASSWORD" -T /usr/bin/codesign
  # The .p12 carries only the leaf; find-identity -v and codesign need the issuing
  # intermediate from a keychain on the search list to validate the chain.
  # Apple's Developer ID G2 CA, SHA-256 F16CD3C54C7F83CEA4BF1A3E6A0819C8AAA8E4A1528FD144715F350643D2DF3A
  intermediate="$(dirname "${BASH_SOURCE[0]}")/apple/DeveloperIDG2CA.cer"
  INTERMEDIATE="$intermediate" python - <<'PY'
import hashlib
import os
from pathlib import Path
expected = "f16cd3c54c7f83cea4bf1a3e6a0819c8aaa8e4a1528fd144715f350643d2df3a"
if hashlib.sha256(Path(os.environ["INTERMEDIATE"]).read_bytes()).hexdigest() != expected:
    raise SystemExit("DeveloperIDG2CA.cer does not match its pinned SHA-256")
PY
  security import "$intermediate" -k "$keychain"
  if [[ ! -f "$search_list" ]]; then
    security list-keychains -d user | sed -e 's/^[[:space:]]*"//' -e 's/"$//' > "$search_list"
  fi
  keychains=()
  while IFS= read -r line; do
    if [[ -n "$line" && "$line" != "$keychain" ]]; then keychains+=("$line"); fi
  done < "$search_list"
  security list-keychains -d user -s "$keychain" ${keychains[@]+"${keychains[@]}"}
  security set-key-partition-list -S apple-tool:,apple:,codesign: -s -k "$keychain_password" "$keychain" >/dev/null
  identity="$(security find-identity -v -p codesigning "$keychain" | sed -n "s/.*\"\(Developer ID Application: .* ($APPLE_TEAM_ID)\)\".*/\1/p")"
  [[ -n "$identity" && "$identity" != *$'\n'* ]] || { echo 'expected one Developer ID Application identity for APPLE_TEAM_ID' >&2; exit 1; }
  codesign --keychain "$keychain" --force --sign "$identity" --options runtime --timestamp ${identifier[@]+"${identifier[@]}"} "${paths[@]}"
  codesign --verify --strict --verbose=2 "${paths[@]}"
  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then printf 'signed=true\nnotarize=%s\n' "$notarize" >> "$GITHUB_OUTPUT"; fi
elif [[ "$mode" == notarize ]]; then
  [[ ${#paths[@]} -eq 1 ]] || { echo 'notarize takes one file or directory' >&2; exit 2; }
  rm -f "$RUNNER_TEMP/rlmesh-notary.zip"
  ditto -c -k --keepParent "${paths[0]}" "$RUNNER_TEMP/rlmesh-notary.zip"
  status=0
  xcrun notarytool submit "$RUNNER_TEMP/rlmesh-notary.zip" --wait --timeout 15m \
    --key "$RUNNER_TEMP/rlmesh-notary.p8" --key-id "$APPLE_NOTARY_KEY_ID" \
    --issuer "$APPLE_NOTARY_ISSUER_ID" --output-format json > "$evidence/notarization.json" || status=$?
  if ! EVIDENCE="$evidence" python - <<'PY'
import json
import os
from pathlib import Path
result = json.loads(Path(os.environ["EVIDENCE"], "notarization.json").read_text())
raise SystemExit(0 if result.get("status") == "Accepted" else 1)
PY
  then
    submission="$(EVIDENCE="$evidence" python - <<'PY'
import json
import os
from pathlib import Path
try:
    print(json.loads(Path(os.environ["EVIDENCE"], "notarization.json").read_text()).get("id", ""))
except (ValueError, OSError):
    print("")
PY
)"
    if [[ -n "$submission" ]]; then
      xcrun notarytool log "$submission" --key "$RUNNER_TEMP/rlmesh-notary.p8" \
        --key-id "$APPLE_NOTARY_KEY_ID" --issuer "$APPLE_NOTARY_ISSUER_ID" \
        "$evidence/notarization-log.json" || true
    fi
    echo '::error::Notarization was not Accepted; see the macos-notarization artifact.' >&2
    exit 1
  fi
  exit "$status"
else
  echo "unknown signing phase: $mode" >&2
  exit 2
fi
