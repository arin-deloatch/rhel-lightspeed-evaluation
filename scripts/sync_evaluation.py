"""Pin the extension to a selected upstream commit, defaulting to its recorded pin."""

import argparse
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from urllib.parse import urldefrag

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "lightspeed-evaluation"
GIT_URL = "https://github.com/lightspeed-core/lightspeed-evaluation.git"


def read_toml(path: Path) -> dict:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def recorded_pin() -> str:
    dependencies = read_toml(ROOT / "pyproject.toml")["project"]["dependencies"]
    pins = [
        match.group(1)
        for dependency in dependencies
        if (
            match := re.fullmatch(
                rf"{PACKAGE}\s*@\s*git\+{re.escape(GIT_URL.removesuffix('.git'))}"
                r"(?:\.git)?@([0-9a-f]{40})",
                dependency,
            )
        )
    ]
    if len(pins) != 1:
        raise ValueError(
            "pyproject.toml must declare exactly one lightspeed-evaluation Git commit pin"
        )
    return pins[0]


def check_project(revision: str) -> None:
    manifest = read_toml(ROOT / "pyproject.toml")
    if manifest["project"]["name"] != "rhel-lightspeed-evaluation":
        raise ValueError("Unexpected project")
    requirement = f"{PACKAGE} @ git+{GIT_URL}@{revision}"
    if requirement not in manifest["project"]["dependencies"]:
        raise ValueError("Package metadata does not pin the selected commit")
    if PACKAGE in manifest.get("tool", {}).get("uv", {}).get("sources", {}):
        raise ValueError("Remove the uv source override; the package dependency carries the pin")
    packages = [p for p in read_toml(ROOT / "uv.lock")["package"] if p["name"] == PACKAGE]
    if len(packages) != 1:
        raise ValueError("Lockfile must contain exactly one lightspeed-evaluation package")
    url, commit = urldefrag(packages[0].get("source", {}).get("git", ""))
    if commit != revision or url.split("?")[0].removesuffix(".git") != GIT_URL.removesuffix(".git"):
        raise ValueError("Lockfile does not resolve the selected commit")


def run(*args: str) -> None:
    env = {
        **os.environ,
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "DEEPEVAL_TELEMETRY_OPT_OUT": "YES",
    }
    env.pop("VIRTUAL_ENV", None)
    subprocess.run(["uv", *args], cwd=ROOT, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--revision", help="Full upstream Git commit SHA (default: pyproject.toml pin)"
    )
    parser.add_argument(
        "--check", action="store_true", help="Check metadata and lock without changes"
    )
    args = parser.parse_args()
    try:
        revision = recorded_pin() if args.revision is None else args.revision
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Revision must be a full 40-character lowercase Git commit SHA")
        if not args.check:
            run("add", "--raw", "--no-sync", f"{PACKAGE} @ git+{GIT_URL}@{revision}")
        check_project(revision)
        if not args.check:
            run("sync", "--locked", "--extra", "dev")
            run("run", "--locked", "pytest", "tests/", "-q")
        print(f"Extension matches selected upstream commit {revision}")
        return 0
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(
            f"Pin synchronization failed: {error}. Review pyproject.toml and uv.lock.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
