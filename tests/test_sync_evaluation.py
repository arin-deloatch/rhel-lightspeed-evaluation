"""The selected pin must survive package installation, not just uv sync."""

import importlib.util
import subprocess
from importlib.metadata import requires
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "extension_sync", Path(__file__).resolve().parents[1] / "scripts/sync_evaluation.py"
)
assert spec is not None and spec.loader is not None
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)
PIN = "a" * 40


@pytest.fixture
def projects(tmp_path, monkeypatch):
    project = tmp_path / "extension"
    project.mkdir()
    requirement = f"lightspeed-evaluation @ git+{sync.GIT_URL}@{PIN}"
    (project / "pyproject.toml").write_text(
        f'[project]\nname = "rhel-lightspeed-evaluation"\ndependencies = ["{requirement}"]\n'
    )
    (project / "uv.lock").write_text(
        f'[[package]]\nname = "lightspeed-evaluation"\n'
        f'source = {{ git = "{sync.GIT_URL}?rev={PIN}#{PIN}" }}\n'
    )
    monkeypatch.setattr(sync, "ROOT", project)
    monkeypatch.setattr("sys.argv", ["sync"])
    return project


@pytest.mark.usefixtures("projects")
def test_read_only_check_and_sync_order(monkeypatch):
    calls = []
    monkeypatch.setattr(sync, "run", lambda *args: calls.append(args))
    argv = sync.sys.argv[:]
    monkeypatch.setattr("sys.argv", [*argv, "--check"])
    assert sync.main() == 0
    assert calls == []
    monkeypatch.setattr("sys.argv", argv)
    assert sync.main() == 0
    assert calls == [
        ("add", "--raw", "--no-sync", f"lightspeed-evaluation @ git+{sync.GIT_URL}@{PIN}"),
        ("sync", "--locked", "--extra", "dev"),
        ("run", "--locked", "pytest", "tests/", "-q"),
    ]


@pytest.mark.parametrize("file", ["pyproject.toml", "uv.lock"])
def test_pin_drift(projects, file):
    path = projects / file
    path.write_text(path.read_text().replace(PIN, "b" * 40))
    with pytest.raises(ValueError, match="selected commit"):
        sync.check_project(PIN)


@pytest.mark.parametrize("revision", ["main", "v0.8.0", "abc123"])
def test_recorded_pin_requires_full_commit(projects, revision):
    path = projects / "pyproject.toml"
    path.write_text(path.read_text().replace(PIN, revision))
    with pytest.raises(ValueError, match="Git commit pin"):
        sync.recorded_pin()


@pytest.mark.usefixtures("projects")
@pytest.mark.parametrize("revision", ["main", "v0.8.0", "abc123", "", "g" * 40])
def test_invalid_override_rejected_before_changes(monkeypatch, revision):
    monkeypatch.setattr("sys.argv", ["sync", "--revision", revision])
    monkeypatch.setattr(sync, "run", lambda *_: pytest.fail("must not change dependencies"))
    assert sync.main() == 1


def test_explicit_override_updates_recorded_default(projects, monkeypatch):
    revision = "b" * 40
    calls = []

    def run(*args):
        calls.append(args)
        if args[0] == "add":
            assert args[-1] == f"lightspeed-evaluation @ git+{sync.GIT_URL}@{revision}"
            for name in ("pyproject.toml", "uv.lock"):
                path = projects / name
                path.write_text(path.read_text().replace(PIN, revision))

    monkeypatch.setattr(sync, "run", run)
    monkeypatch.setattr("sys.argv", ["sync", "--revision", revision, "--check"])
    assert sync.main() == 1
    assert calls == []
    monkeypatch.setattr("sys.argv", ["sync", "--revision", revision])
    assert sync.main() == 0
    assert sync.recorded_pin() == revision
    calls.clear()
    monkeypatch.setattr("sys.argv", ["sync", "--check"])
    assert sync.main() == 0
    assert calls == []


@pytest.mark.usefixtures("projects")
def test_test_failure_propagates(monkeypatch):
    def fail(*args):
        if "pytest" in args:
            raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(sync, "run", fail)
    assert sync.main() == 1


def test_installed_metadata_carries_the_upstream_commit():
    from packaging.requirements import Requirement

    manifest = sync.read_toml(Path(__file__).resolve().parents[1] / "pyproject.toml")
    declared = next(
        Requirement(item)
        for item in manifest["project"]["dependencies"]
        if Requirement(item).name == sync.PACKAGE
    )
    installed = next(
        Requirement(item)
        for item in requires("rhel-lightspeed-evaluation")
        if Requirement(item).name == sync.PACKAGE
    )
    assert installed.url == declared.url
    assert installed.url.startswith(f"git+{sync.GIT_URL}@")
