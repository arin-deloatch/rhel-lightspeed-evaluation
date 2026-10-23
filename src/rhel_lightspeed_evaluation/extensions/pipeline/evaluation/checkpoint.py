"""Atomic conversation checkpoints for local, file-backed evaluations."""

import hashlib
import json
import os
from importlib.metadata import distribution
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, TextIO

from lightspeed_evaluation.core.models import EvaluationData, EvaluationResult, SystemConfig
from lightspeed_evaluation.core.storage import FileBackendConfig
from pydantic import BaseModel, ConfigDict


class Manifest(BaseModel):
    """Store a fingerprint, not potentially secret configuration values."""

    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    fingerprint: str


class ConversationCheckpoint(BaseModel):
    """Results and amended input become durable in the same atomic write."""

    model_config = ConfigDict(extra="forbid")
    conversation: EvaluationData
    results: list[EvaluationResult]


def _canonical(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _canonical(value.model_dump(mode="python"))
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, set | frozenset):
        return sorted((_canonical(item) for item in value), key=lambda item: json.dumps(item))
    if isinstance(value, list | tuple):
        return [_canonical(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(_canonical(value), sort_keys=True).encode()).hexdigest()


def validate_storage(config: SystemConfig) -> None:
    """Replaying results into external backends could duplicate their records."""
    if any(not isinstance(item, FileBackendConfig) for item in config.storage):
        raise ValueError("Resume requires file-only storage; external backends are not supported")


class CheckpointStore:
    """One writer per directory; a killed writer releases the OS lock automatically."""

    def __init__(self, directory: Path):
        self.directory = directory
        self._lock: TextIO | None = None

    def __enter__(self) -> "CheckpointStore":
        try:
            import fcntl
        except ImportError:
            raise ValueError("Resume requires Linux/macOS file locking") from None
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = (self.directory / "lock").open("a")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            self._lock = None
            raise ValueError("Another evaluation is using this checkpoint directory") from None
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._lock is not None:
            self._lock.close()
            self._lock = None

    def _write(self, path: Path, model: BaseModel) -> None:
        # Rename is the commit point: a crash leaves either the old file or the whole new one.
        temporary = path.with_suffix(".tmp")
        with temporary.open("w") as stream:
            stream.write(model.model_dump_json())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def validate(
        self, config: SystemConfig, data: list[EvaluationData], metadata: Any = None
    ) -> None:
        """Refuse changed inputs, configuration or framework versions before reusing work."""
        ids = [conversation.conversation_group_id for conversation in data]
        if len(set(ids)) != len(ids):
            raise ValueError("Resume requires unique conversation_group_id values")
        runtime = {}
        for name in ("rhel-lightspeed-evaluation", "lightspeed-evaluation", "deepeval", "ragas"):
            package = distribution(name)
            runtime[name] = (package.version, package.read_text("direct_url.json"))
        script_paths = {
            path
            for conversation in data
            for path in (
                conversation.setup_script,
                conversation.cleanup_script,
                *(turn.verify_script for turn in conversation.turns),
            )
            if path is not None
        }
        scripts = {
            str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in script_paths
        }
        manifest = Manifest(fingerprint=_digest([config, data, metadata, runtime, scripts]))
        path = self.directory / "manifest.json"
        if path.exists():
            if Manifest.model_validate_json(path.read_text()) != manifest:
                raise ValueError(
                    "Checkpoint does not match the inputs, configuration or dependencies; "
                    "use a fresh output directory"
                )
        else:
            if list(self.directory.glob("conversation-*.json")):
                raise ValueError("Checkpoint manifest is missing; refusing to reuse saved results")
            self._write(path, manifest)

    def load(self, data: list[EvaluationData]) -> dict[str, ConversationCheckpoint]:
        expected = {conversation.conversation_group_id for conversation in data}
        saved = {}
        for path in self.directory.glob("conversation-*.json"):
            try:
                record = ConversationCheckpoint.model_validate_json(path.read_text())
            except ValueError:
                raise ValueError(f"Unreadable conversation checkpoint: {path.name}") from None
            identifier = record.conversation.conversation_group_id
            if (
                identifier not in expected
                or path != self._path(identifier)
                or any(result.conversation_group_id != identifier for result in record.results)
            ):
                raise ValueError("Invalid conversation checkpoint; refusing to resume")
            saved[identifier] = record
        return saved

    def _path(self, identifier: str) -> Path:
        return self.directory / f"conversation-{_digest(identifier)}.json"

    def save(self, conversation: EvaluationData, results: list[EvaluationResult]) -> None:
        self._write(
            self._path(conversation.conversation_group_id),
            ConversationCheckpoint(conversation=conversation, results=results),
        )
