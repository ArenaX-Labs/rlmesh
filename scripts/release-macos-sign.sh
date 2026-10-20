#!/usr/bin/env bash
# Never enable tracing. Credentials exist only on the ephemeral release runner.
set -euo pipefail
: "${RUNNER_TEMP:?release runner scratch is required}"
keychain="$RUNNER_TEMP/rlmesh-signing.keychain-db"
library="target/macos-universal/librlmesh_capi.dylib"
mode="${1:-sign}"
if [[ "$mode" == cleanup ]]; then
  security delete-keychain "$keychain" 2>/dev/null || true
  rm -f "$RUNNER_TEMP/rlmesh-signing.p12" "$RUNNER_TEMP/rlmesh-notary.p8" "$RUNNER_TEMP/rlmesh-notary.zip"
  exit 0
fi

if [[ "$mode" == sign ]]; then
  rm -f target/macos-universal/notarization.json target/macos-universal/notarization-log.json
  missing=false
  for secret in APPLE_DEVELOPER_ID_P12 APPLE_DEVELOPER_ID_P12_PASSWORD APPLE_TEAM_ID APPLE_NOTARY_KEY APPLE_NOTARY_KEY_ID APPLE_NOTARY_ISSUER_ID; do
    if [[ -z "${!secret:-}" ]]; then missing=true; fi
  done
  if [[ "$missing" == true ]]; then
    if [[ "${GITHUB_REF:-}" == refs/tags/v* && "${RLMESH_ALLOW_UNSIGNED_MACOS:-}" != true ]]; then
      echo '::error::Apple signing/notarization secrets are incomplete; tag releases require them unless RLMESH_ALLOW_UNSIGNED_MACOS is explicitly true.' >&2
      exit 1
    fi
    echo '::warning::Apple secrets are incomplete; explicitly allowing an ad-hoc signed macOS library without notarization.'
    codesign --force --sign - "$library"
    codesign --verify --strict --verbose=2 "$library"
    if [[ -n "${GITHUB_OUTPUT:-}" ]]; then echo 'notarize=false' >> "$GITHUB_OUTPUT"; fi
    exit 0
  fi
  python - <<'PY'
import base64
import os
from pathlib import Path
root = Path(os.environ["RUNNER_TEMP"])
for name, contents in (("rlmesh-signing.p12", base64.b64decode("".join(os.environ["APPLE_DEVELOPER_ID_P12"].split()), validate=True)),
                       ("rlmesh-notary.p8", os.environ["APPLE_NOTARY_KEY"].encode())):
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
  security set-key-partition-list -S apple-tool:,apple:,codesign: -s -k "$keychain_password" "$keychain" >/dev/null
  identity="$(security find-identity -v -p codesigning "$keychain" | sed -n "s/.*\"\(Developer ID Application: .* ($APPLE_TEAM_ID)\)\".*/\1/p")"
  [[ -n "$identity" && "$identity" != *$'\n'* ]] || { echo 'expected one Developer ID Application identity for APPLE_TEAM_ID' >&2; exit 1; }
  codesign --keychain "$keychain" --force --sign "$identity" --options runtime --timestamp "$library"
  codesign --verify --strict --verbose=2 "$library"
  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then echo 'notarize=true' >> "$GITHUB_OUTPUT"; fi
elif [[ "$mode" == notarize ]]; then
  ditto -c -k --keepParent "$library" "$RUNNER_TEMP/rlmesh-notary.zip"
  # Preserve the response even when notarytool exits nonzero (Invalid/timeout).
  status=0
  xcrun notarytool submit "$RUNNER_TEMP/rlmesh-notary.zip" --wait --timeout 15m \
    --key "$RUNNER_TEMP/rlmesh-notary.p8" --key-id "$APPLE_NOTARY_KEY_ID" \
    --issuer "$APPLE_NOTARY_ISSUER_ID" --output-format json > target/macos-universal/notarization.json || status=$?
  if ! python - <<'PY'
import json
from pathlib import Path
result = json.loads(Path("target/macos-universal/notarization.json").read_text())
raise SystemExit(0 if result.get("status") == "Accepted" else 1)
PY
  then
    submission="$(python - <<'PY'
import json
from pathlib import Path
try:
    print(json.loads(Path("target/macos-universal/notarization.json").read_text()).get("id", ""))
except (ValueError, OSError):
    print("")
PY
)"
    if [[ -n "$submission" ]]; then
      xcrun notarytool log "$submission" --key "$RUNNER_TEMP/rlmesh-notary.p8" \
        --key-id "$APPLE_NOTARY_KEY_ID" --issuer "$APPLE_NOTARY_ISSUER_ID" \
        target/macos-universal/notarization-log.json || true
    fi
    echo '::error::Notarization was not Accepted; see the macos-notarization artifact.' >&2
    exit 1
  fi
  exit "$status"
else
  echo "unknown signing phase: $mode" >&2
  exit 2
fi
