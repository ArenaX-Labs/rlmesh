"""Exercise macOS signing policy on Linux with mocked Apple commands."""

import os
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SECRETS = (
    "APPLE_DEVELOPER_ID_P12",
    "APPLE_DEVELOPER_ID_P12_PASSWORD",
    "APPLE_TEAM_ID",
    "APPLE_NOTARY_KEY",
    "APPLE_NOTARY_KEY_ID",
    "APPLE_NOTARY_ISSUER_ID",
)


class SigningTests(unittest.TestCase):
    def setUp(self):
        scratch = pathlib.Path.home() / ".cache/rlmesh-release/tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        (self.root / "target/macos-universal").mkdir(parents=True)
        (self.root / "bin").mkdir()
        self.env = dict(
            os.environ,
            RUNNER_TEMP=str(self.root),
            GITHUB_REF="refs/tags/v1.2.3",
            GITHUB_OUTPUT=str(self.root / "output"),
            RLMESH_ALLOW_UNSIGNED_MACOS="",
            PATH=str(self.root / "bin") + os.pathsep + os.environ["PATH"],
        )
        for secret in SECRETS:
            self.env[secret] = ""
        for command in ("codesign", "security", "ditto", "xcrun"):
            path = self.root / "bin" / command
            path.write_text("""#!/usr/bin/env bash
printf '%s\\n' "$(basename "$0") $*" >> "$RUNNER_TEMP/commands"
case "$(basename "$0") $*" in
  "security find-identity"*) echo '1) ABC "Developer ID Application: Fixture (TEAM)"' ;;
  "xcrun notarytool submit"*) echo '{"id":"fixture-submission","status":"Invalid"}'; exit 1 ;;
  "xcrun notarytool log"*) echo '{"fixture":"failure log"}' > "${!#}" ;;
esac
""")
            path.chmod(0o700)

    def invoke(self, mode="sign"):
        return subprocess.run(
            ["bash", str(ROOT / "scripts/release-macos-sign.sh"), mode],
            cwd=self.root,
            env=self.env,
            text=True,
            capture_output=True,
            timeout=10,
        )

    def credentials(self):
        self.env.update(dict.fromkeys(SECRETS, "fixture"))
        self.env["APPLE_DEVELOPER_ID_P12"] = "Zml4dHVyZQ=="
        self.env["APPLE_TEAM_ID"] = "TEAM"

    def test_each_missing_secret_fails_tag_without_override(self):
        for secret in SECRETS:
            self.credentials()
            self.env[secret] = ""
            result = self.invoke()
            self.assertNotEqual(result.returncode, 0, secret)
            self.assertIn("::error::", result.stderr)
        self.assertFalse((self.root / "commands").exists())

    def test_explicit_override_ad_hoc_signs_and_skips_notary(self):
        self.env["RLMESH_ALLOW_UNSIGNED_MACOS"] = "true"
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("::warning::", result.stdout)
        self.assertIn("codesign --force --sign -", (self.root / "commands").read_text())
        self.assertIn("notarize=false", (self.root / "output").read_text())

    def test_keys_are_exclusive_private_and_invalid_submission_gets_log(self):
        self.credentials()
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in ("rlmesh-signing.p12", "rlmesh-notary.p8"):
            self.assertEqual((self.root / name).stat().st_mode & 0o777, 0o600)
        self.assertNotEqual(
            self.invoke().returncode, 0, "existing key files must be refused"
        )
        result = self.invoke("notarize")
        self.assertNotEqual(result.returncode, 0)
        calls = (self.root / "commands").read_text()
        self.assertIn("--timeout 15m", calls)
        self.assertIn("notarytool log fixture-submission", calls)
        self.assertTrue(
            (self.root / "target/macos-universal/notarization-log.json").exists()
        )
        self.assertEqual(self.invoke("cleanup").returncode, 0)
        self.assertFalse((self.root / "rlmesh-notary.p8").exists())


if __name__ == "__main__":
    unittest.main()
