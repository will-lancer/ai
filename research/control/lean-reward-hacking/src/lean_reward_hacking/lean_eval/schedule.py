"""Deterministic factorial schedule for the Lean proof-interface pilot.

The scheduler has no clock, filesystem, random module, network, or provider
side effects in its identity path.  It accepts only trusted task hashes and
public model configuration, creates every assignment before dispatch, and
orders each model-repeat cell with SHA-256.  The twelve cells are then
interleaved round-robin so each consecutive block contains one row from every
cell.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .canonical import (
    CanonicalJSONError,
    CanonicalizationError,
    canonical_bytes,
    canonical_dumps,
    domain_hash,
    hash_config,
    strict_loads,
)
from .conditions import (
    CONDITION_IDS,
    CONDITIONS,
    ConditionError,
    ConditionSpec,
    get_condition,
    resolve_conditions,
)
from .records import assignment_id, pair_id, schedule_id as record_schedule_id, session_id


SCHEMA_VERSION = "lean-eval/schedule-v1"
SCHEDULE_ALGORITHM = "balanced-sha256-v1"
DEFAULT_MASTER_SEED = 17_290_427
DEFAULT_REPEATS: tuple[int, int, int] = (1, 2, 3)
DEFAULT_MODES: tuple[str, str] = ("whole_file", "frozen_hole")
MODE_ORDER: Mapping[str, int] = {mode: index for index, mode in enumerate(DEFAULT_MODES)}

_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")
_PROVIDER_MODELS: Mapping[str, frozenset[str]] = {
    "openai": frozenset({"gpt-5.6-sol", "gpt-5.6-luna"}),
    "anthropic": frozenset({"claude-opus-5"}),
    "google": frozenset({"gemini-3.7-flash"}),
    "mock": frozenset({"mock"}),
}


DEFAULT_MODEL_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "id": "openai_sol",
        "provider": "openai",
        "wire_model_id": "gpt-5.6-sol",
        "reasoning": "medium",
        "settings": {"effort": "medium"},
    },
    {
        "id": "anthropic_opus5",
        "provider": "anthropic",
        "wire_model_id": "claude-opus-5",
        "reasoning": "medium",
        "settings": {"thinking": "adaptive", "effort": "medium"},
    },
    {
        "id": "google_flash37",
        "provider": "google",
        "wire_model_id": "gemini-3.7-flash",
        "reasoning": "medium",
        "settings": {"thinking": "medium"},
    },
    {
        "id": "openai_luna",
        "provider": "openai",
        "wire_model_id": "gpt-5.6-luna",
        "reasoning": "medium",
        "settings": {"effort": "medium"},
    },
)


class ScheduleError(ValueError):
    """Raised when trusted schedule inputs violate the pilot contract."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ScheduleError(f"{name} must be a non-empty string")
    if "\0" in value or "\r" in value:
        raise ScheduleError(f"{name} contains a prohibited control character")
    try:
        value.encode("utf-8", "strict")
    except UnicodeEncodeError as exc:
        raise ScheduleError(f"{name} is not valid UTF-8") from exc
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ScheduleError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ScheduleError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class TaskRef:
    """Public task identity used by the schedule."""

    id: str
    task_hash: str

    def __post_init__(self) -> None:
        _text(self.id, "task id")
        _digest(self.task_hash, "task_hash")

    @property
    def task_id(self) -> str:
        return self.id

    def to_record(self) -> dict[str, str]:
        return {"id": self.id, "task_hash": self.task_hash}


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Provider configuration that is safe to expose in a schedule row."""

    id: str
    provider: str
    wire_model_id: str
    reasoning: str = "medium"
    settings: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        _text(self.id, "model id")
        _text(self.provider, "provider")
        _text(self.wire_model_id, "wire_model_id")
        _text(self.reasoning, "reasoning")
        if self.provider in _PROVIDER_MODELS and self.wire_model_id not in _PROVIDER_MODELS[self.provider]:
            raise ScheduleError(
                f"wire model {self.wire_model_id!r} is not approved for provider {self.provider!r}"
            )
        if not isinstance(self.settings, Mapping):
            raise ScheduleError("model settings must be an object")
        try:
            checked = json.loads(canonical_dumps(dict(self.settings)))
        except (CanonicalizationError, TypeError, ValueError) as exc:
            raise ScheduleError("model settings are not canonical JSON") from exc
        object.__setattr__(self, "settings", MappingProxyType(checked))

    @property
    def model_id(self) -> str:
        return self.id

    @property
    def model(self) -> str:
        return self.wire_model_id

    @property
    def reasoning_settings(self) -> Mapping[str, Any]:
        return self.settings

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "provider": self.provider,
            "wire_model_id": self.wire_model_id,
            "reasoning": self.reasoning,
            "settings": dict(self.settings),
            "config_hash": hash_config(
                {
                    "id": self.id,
                    "provider": self.provider,
                    "wire_model_id": self.wire_model_id,
                    "reasoning": self.reasoning,
                    "settings": dict(self.settings),
                }
            ),
        }


@dataclass(frozen=True, slots=True)
class Assignment:
    """One row in the pre-dispatch factorial manifest."""

    order: int
    cell_index: int
    cell_position: int
    cell_id: str
    task_id: str
    task_hash: str
    model_id: str
    provider: str
    wire_model_id: str
    reasoning: str
    reasoning_settings: Mapping[str, Any]
    condition: str
    condition_hash: str
    mode: str
    repeat: int
    assignment_id: str
    pair_id: str
    session_id: str
    schedule_id: str = ""

    def __post_init__(self) -> None:
        _int(self.order, "order")
        _int(self.cell_index, "cell_index")
        _int(self.cell_position, "cell_position")
        _text(self.cell_id, "cell_id")
        _text(self.task_id, "task_id")
        _digest(self.task_hash, "task_hash")
        _text(self.model_id, "model_id")
        _text(self.provider, "provider")
        _text(self.wire_model_id, "wire_model_id")
        _text(self.reasoning, "reasoning")
        _text(self.condition, "condition")
        _digest(self.condition_hash, "condition_hash")
        if self.mode not in MODE_ORDER:
            raise ScheduleError(f"unknown mode: {self.mode!r}")
        _int(self.repeat, "repeat")
        for name, value in (
            ("assignment_id", self.assignment_id),
            ("pair_id", self.pair_id),
            ("session_id", self.session_id),
        ):
            _digest(value, name)
        if not isinstance(self.reasoning_settings, Mapping):
            raise ScheduleError("reasoning_settings must be an object")

    @property
    def index(self) -> int:
        """One-based external row number, while ``order`` stays zero-based."""

        return self.order + 1

    def to_record(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "index": self.index,
            "cell_index": self.cell_index,
            "cell_position": self.cell_position,
            "cell_id": self.cell_id,
            "task_id": self.task_id,
            "task_hash": self.task_hash,
            "model_id": self.model_id,
            "provider": self.provider,
            "wire_model_id": self.wire_model_id,
            "reasoning": self.reasoning,
            "reasoning_settings": dict(self.reasoning_settings),
            "condition": self.condition,
            "condition_hash": self.condition_hash,
            "mode": self.mode,
            "repeat": self.repeat,
            "assignment_id": self.assignment_id,
            "pair_id": self.pair_id,
            "session_id": self.session_id,
            "schedule_id": self.schedule_id,
        }


@dataclass(frozen=True, slots=True)
class Schedule:
    """Immutable schedule and its complete manifest metadata."""

    assignments: tuple[Assignment, ...]
    master_seed: int
    algorithm: str
    schedule_id: str
    tasks: tuple[TaskRef, ...]
    models: tuple[ModelConfig, ...]
    conditions: tuple[ConditionSpec, ...]
    modes: tuple[str, ...]
    repeats: tuple[int, ...]
    run_id: str = ""
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ScheduleError("unsupported schedule schema")
        _int(self.master_seed, "master_seed")
        if self.algorithm != SCHEDULE_ALGORITHM:
            raise ScheduleError("unsupported schedule algorithm")
        if self.run_id:
            _text(self.run_id, "run_id")
        _digest(self.schedule_id, "schedule_id")
        if len(self.assignments) != len(self.tasks) * len(self.models) * len(self.conditions) * len(self.modes) * len(self.repeats):
            raise ScheduleError("assignment count does not match the factorial inputs")

    def __iter__(self) -> Iterator[Assignment]:
        return iter(self.assignments)

    def __len__(self) -> int:
        return len(self.assignments)

    def __getitem__(self, index: int | slice) -> Assignment | tuple[Assignment, ...]:
        return self.assignments[index]

    @property
    def assignment_ids(self) -> tuple[str, ...]:
        return tuple(row.assignment_id for row in self.assignments)

    @property
    def pair_ids(self) -> frozenset[str]:
        return frozenset(row.pair_id for row in self.assignments)

    @property
    def cell_count(self) -> int:
        return len(self.models) * len(self.repeats)

    @property
    def cell_size(self) -> int:
        return len(self.tasks) * len(self.conditions) * len(self.modes)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "algorithm": self.algorithm,
            "master_seed": self.master_seed,
            "schedule_id": self.schedule_id,
            "run_id": self.run_id,
            "task_count": len(self.tasks),
            "model_count": len(self.models),
            "condition_count": len(self.conditions),
            "mode_count": len(self.modes),
            "repeat_count": len(self.repeats),
            "assignment_count": len(self.assignments),
            "pair_count": len(self.pair_ids),
            "cell_count": self.cell_count,
            "cell_size": self.cell_size,
            "tasks": [task.to_record() for task in self.tasks],
            "models": [model.to_record() for model in self.models],
            "conditions": [condition.to_record() for condition in self.conditions],
            "modes": list(self.modes),
            "repeats": list(self.repeats),
            "assignments": [row.to_record() for row in self.assignments],
        }

    def to_json_bytes(self) -> bytes:
        return canonical_bytes(self.to_manifest()) + b"\n"

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.to_json_bytes())
        return target


def _task_input(value: Any) -> Iterable[Any]:
    if value is None:
        try:
            from .corpus import load_catalog

            return load_catalog().tasks
        except Exception as exc:
            raise ScheduleError("tasks are required when the trusted catalog is unavailable") from exc
    if isinstance(value, Mapping) and "tasks" in value:
        return _task_input(value["tasks"])
    if hasattr(value, "tasks") and not isinstance(value, (str, bytes, bytearray)):
        return _task_input(getattr(value, "tasks"))
    if isinstance(value, (str, bytes, bytearray)):
        raise ScheduleError("tasks must be an iterable of task objects")
    try:
        return value
    except TypeError as exc:
        raise ScheduleError("tasks must be an iterable") from exc


def normalize_tasks(value: Any = None) -> tuple[TaskRef, ...]:
    """Extract only public task IDs and trusted task hashes."""

    try:
        records = list(_task_input(value))
    except TypeError as exc:
        raise ScheduleError("tasks must be an iterable") from exc
    result: list[TaskRef] = []
    for index, item in enumerate(records):
        if isinstance(item, TaskRef):
            task = item
        elif isinstance(item, Mapping):
            task_id = item.get("id", item.get("task_id"))
            task_hash = item.get("task_hash", item.get("hash"))
            if task_id is None or task_hash is None:
                raise ScheduleError(f"task {index} lacks id or task_hash")
            task = TaskRef(str(task_id), task_hash)
        else:
            task_id = getattr(item, "id", getattr(item, "task_id", None))
            task_hash = getattr(item, "task_hash", getattr(item, "hash", None))
            if task_id is None or task_hash is None:
                raise ScheduleError(f"task {index} lacks id or task_hash")
            task = TaskRef(str(task_id), task_hash)
        result.append(task)
    if not result:
        raise ScheduleError("at least one task is required")
    result.sort(key=lambda item: item.id)
    if len({item.id for item in result}) != len(result):
        raise ScheduleError("task IDs must be unique")
    if len({item.task_hash for item in result}) != len(result):
        raise ScheduleError("task hashes must be unique")
    return tuple(result)


def _model_records(value: Any) -> list[tuple[str | None, Any]]:
    if value is None:
        return [(item["id"], item) for item in DEFAULT_MODEL_CONFIGS]
    if isinstance(value, Mapping):
        if any(key in value for key in ("provider", "model", "model_id", "wire_model_id")):
            return [(None, value)]
        return [(str(key), item) for key, item in value.items()]
    if isinstance(value, (str, bytes, bytearray)):
        raise ScheduleError("models must be an iterable of configurations")
    try:
        return [(None, item) for item in value]
    except TypeError as exc:
        raise ScheduleError("models must be an iterable") from exc


def normalize_models(value: Any = None) -> tuple[ModelConfig, ...]:
    """Normalize provider configs while rejecting moving aliases and duplicates."""

    result: list[ModelConfig] = []
    for key, item in _model_records(value):
        if isinstance(item, ModelConfig):
            config = item
        elif isinstance(item, Mapping):
            model_id = item.get("id", item.get("key", item.get("name", key)))
            provider = item.get("provider")
            wire = item.get("wire_model_id", item.get("wire_id", item.get("model")))
            if wire is None and item.get("model_id") is not None:
                wire = item["model_id"]
            if model_id is None:
                model_id = item.get("model_id", wire)
            reasoning = item.get("reasoning")
            settings = item.get("settings")
            if settings is None:
                settings = {}
                for field in ("effort", "thinking"):
                    if field in item:
                        settings[field] = item[field]
            if reasoning is None:
                reasoning = item.get("effort", "medium")
            if provider is None or wire is None or model_id is None:
                raise ScheduleError("model config requires id, provider, and wire_model_id")
            config = ModelConfig(str(model_id), str(provider), str(wire), str(reasoning), settings)
        else:
            raise ScheduleError("model config must be a mapping or ModelConfig")
        result.append(config)
    if not result:
        raise ScheduleError("at least one model config is required")
    result.sort(key=lambda item: (item.id, item.provider, item.wire_model_id))
    ids = [item.id for item in result]
    if len(set(ids)) != len(ids):
        raise ScheduleError("model IDs must be unique")
    wire_ids = [(item.provider, item.wire_model_id) for item in result]
    if len(set(wire_ids)) != len(wire_ids):
        raise ScheduleError("provider/wire model pairs must be unique")
    return tuple(result)


def normalize_modes(value: Iterable[str] | None = None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_MODES
    if isinstance(value, str):
        values = [value]
    else:
        try:
            values = list(value)
        except TypeError as exc:
            raise ScheduleError("modes must be an iterable") from exc
    if not values or any(item not in MODE_ORDER for item in values):
        raise ScheduleError("modes must contain whole_file and/or frozen_hole")
    if len(set(values)) != len(values):
        raise ScheduleError("modes must be unique")
    return tuple(sorted(values, key=MODE_ORDER.__getitem__))


def normalize_repeats(value: Iterable[int] | int | None = None) -> tuple[int, ...]:
    if value is None:
        return DEFAULT_REPEATS
    if isinstance(value, bool):
        raise ScheduleError("repeat count cannot be boolean")
    if isinstance(value, int):
        if value < 1:
            raise ScheduleError("repeat count must be positive")
        values = list(range(1, value + 1))
    else:
        try:
            values = list(value)
        except TypeError as exc:
            raise ScheduleError("repeats must be an iterable or count") from exc
    if not values or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in values):
        raise ScheduleError("repeats must be positive one-based integer indices")
    if len(set(values)) != len(values):
        raise ScheduleError("repeats must be unique")
    return tuple(sorted(values))


def schedule_rank(
    master_seed: int,
    cell_id: str,
    assignment: str | Mapping[str, Any],
    *,
    algorithm: str = SCHEDULE_ALGORITHM,
) -> str:
    """Return the deterministic rank digest used inside one cell.

    The input may be an assignment ID or a row mapping.  Row mappings are
    reduced to the stable assignment ID before hashing, so operational fields
    cannot perturb ordering.
    """

    master_seed = _int(master_seed, "master_seed")
    if algorithm != SCHEDULE_ALGORITHM:
        raise ScheduleError("unsupported schedule algorithm")
    if isinstance(assignment, str):
        assignment_id_value = _digest(assignment, "assignment_id")
    elif isinstance(assignment, Mapping):
        assignment_id_value = assignment.get("assignment_id")
        if not isinstance(assignment_id_value, str):
            raise ScheduleError("assignment row lacks assignment_id")
        _digest(assignment_id_value, "assignment_id")
    else:
        raise ScheduleError("assignment must be an ID or mapping")
    cell_id = _text(cell_id, "cell_id")
    return domain_hash(
        "lean-eval/schedule-rank",
        {
            "algorithm": algorithm,
            "master_seed": master_seed,
            "cell_id": cell_id,
            "assignment_id": assignment_id_value,
        },
    )


def _assignment_sort_key(seed: int, algorithm: str, cell_id: str, row: Mapping[str, Any]) -> tuple[str, str]:
    # The assignment ID is a tie-breaker even though SHA-256 collisions are
    # not expected, making the ordering total for synthetic test fixtures too.
    return schedule_rank(seed, cell_id, row, algorithm=algorithm), str(row["assignment_id"])


def _round_robin(cells: Sequence[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    if not cells:
        return []
    result: list[dict[str, Any]] = []
    max_size = max(len(cell) for cell in cells)
    for position in range(max_size):
        for cell in cells:
            if position < len(cell):
                result.append(cell[position])
    return result


def _condition_records(values: Iterable[str | ConditionSpec | Mapping[str, Any]] | None) -> tuple[ConditionSpec, ...]:
    try:
        result = resolve_conditions(values)
    except ConditionError as exc:
        raise ScheduleError(str(exc)) from exc
    return result


def build_schedule(
    tasks: Any = None,
    models: Any = None,
    *,
    conditions: Iterable[str | ConditionSpec | Mapping[str, Any]] | None = None,
    modes: Iterable[str] | None = None,
    repeats: Iterable[int] | int | None = None,
    repeat_count: int | None = None,
    master_seed: int = DEFAULT_MASTER_SEED,
    algorithm: str = SCHEDULE_ALGORITHM,
    run_id: str = "",
    require_pilot: bool = False,
    task_manifest: Any = None,
    model_configs: Any = None,
) -> Schedule:
    """Build the complete schedule before any provider dispatch.

    ``task_manifest`` and ``model_configs`` are compatibility keyword names.
    If both aliases are supplied, the primary positional spelling wins only
    when the alias is ``None``; conflicting values are rejected.
    """

    if task_manifest is not None:
        if tasks is not None:
            raise ScheduleError("supply tasks or task_manifest, not both")
        tasks = task_manifest
    if model_configs is not None:
        if models is not None:
            raise ScheduleError("supply models or model_configs, not both")
        models = model_configs
    if repeat_count is not None:
        if repeats is not None:
            raise ScheduleError("supply repeats or repeat_count, not both")
        repeats = repeat_count
    master_seed = _int(master_seed, "master_seed")
    _text(run_id, "run_id") if run_id else None
    if algorithm != SCHEDULE_ALGORITHM:
        raise ScheduleError(f"unsupported schedule algorithm: {algorithm!r}")
    task_refs = normalize_tasks(tasks)
    model_refs = normalize_models(models)
    condition_refs = _condition_records(conditions)
    mode_refs = normalize_modes(modes)
    repeat_refs = normalize_repeats(repeats)
    if require_pilot:
        if len(task_refs) != 8 or len(model_refs) != 4 or condition_refs != CONDITIONS or mode_refs != DEFAULT_MODES or repeat_refs != DEFAULT_REPEATS:
            raise ScheduleError("pilot schedule must be 8 x 2 x 2 x 4 x 3")
        expected_models = {item["wire_model_id"] for item in DEFAULT_MODEL_CONFIGS}
        if {item.wire_model_id for item in model_refs} != expected_models:
            raise ScheduleError("pilot schedule model set is not the frozen four-model set")

    # Build by canonical model/repeat cells.  The inner rows are intentionally
    # generated from canonical task, condition, and mode order; sorting below
    # makes task input order irrelevant.
    cells: list[list[dict[str, Any]]] = []
    cell_meta: list[tuple[str, int, ModelConfig, int]] = []
    cell_index = 0
    for model in model_refs:
        for repeat in repeat_refs:
            cell_id = f"{model.id}/repeat-{repeat}"
            rows: list[dict[str, Any]] = []
            for task in task_refs:
                for condition in condition_refs:
                    for mode in mode_refs:
                        aid = assignment_id(
                            task_id=task.id,
                            task_hash=task.task_hash,
                            model_id=model.id,
                            provider=model.provider,
                            wire_model_id=model.wire_model_id,
                            reasoning=model.reasoning,
                            condition=condition.id,
                            mode=mode,
                            repeat=repeat,
                            run_id=run_id,
                        )
                        pid = pair_id(
                            task_id=task.id,
                            task_hash=task.task_hash,
                            model_id=model.id,
                            provider=model.provider,
                            wire_model_id=model.wire_model_id,
                            reasoning=model.reasoning,
                            condition=condition.id,
                            repeat=repeat,
                            run_id=run_id,
                        )
                        sid = session_id(
                            task_id=task.id,
                            task_hash=task.task_hash,
                            model_id=model.id,
                            provider=model.provider,
                            wire_model_id=model.wire_model_id,
                            reasoning=model.reasoning,
                            condition=condition.id,
                            mode=mode,
                            repeat=repeat,
                            run_id=run_id,
                        )
                        rows.append(
                            {
                                "assignment_id": aid,
                                "pair_id": pid,
                                "session_id": sid,
                                "task_id": task.id,
                                "task_hash": task.task_hash,
                                "model_id": model.id,
                                "provider": model.provider,
                                "wire_model_id": model.wire_model_id,
                                "reasoning": model.reasoning,
                                "reasoning_settings": dict(model.settings),
                                "condition": condition.id,
                                "condition_hash": condition.hash,
                                "mode": mode,
                                "repeat": repeat,
                            }
                        )
            rows.sort(key=lambda row: _assignment_sort_key(master_seed, algorithm, cell_id, row))
            for position, row in enumerate(rows):
                row["cell_id"] = cell_id
                row["cell_index"] = cell_index
                row["cell_position"] = position
            cells.append(rows)
            cell_meta.append((cell_id, cell_index, model, repeat))
            cell_index += 1

    ordered_rows = _round_robin(cells)
    ordered_ids = [str(row["assignment_id"]) for row in ordered_rows]
    sid = record_schedule_id(master_seed, ordered_ids, algorithm=algorithm, run_id=run_id)
    assignments: list[Assignment] = []
    for order, row in enumerate(ordered_rows):
        row["order"] = order
        row["schedule_id"] = sid
        assignments.append(Assignment(**row))
    result = Schedule(
        assignments=tuple(assignments),
        master_seed=master_seed,
        algorithm=algorithm,
        schedule_id=sid,
        tasks=task_refs,
        models=model_refs,
        conditions=condition_refs,
        modes=mode_refs,
        repeats=repeat_refs,
        run_id=run_id,
    )
    validate_schedule(result)
    return result


def build_pilot_schedule(
    tasks: Any = None,
    models: Any = None,
    *,
    run_id: str = "",
    master_seed: int = DEFAULT_MASTER_SEED,
) -> Schedule:
    """Build the fixed 384-row pilot schedule with cardinality checks."""

    return build_schedule(
        tasks,
        models,
        run_id=run_id,
        master_seed=master_seed,
        require_pilot=True,
    )


def write_schedule(schedule: Schedule, path: str | Path) -> Path:
    """Persist a precomputed manifest without changing its identity."""

    if not isinstance(schedule, Schedule):
        raise TypeError("schedule must be a Schedule")
    return schedule.write(path)


def validate_schedule(schedule: Schedule) -> bool:
    """Check all row identities and ordering invariants of a schedule.

    This validation is intentionally independent of the persisted JSON
    representation.  It catches a changed task hash, provider setting, mode,
    repeat, condition hash, row ID, or schedule ID before dispatch.
    """

    if not isinstance(schedule, Schedule):
        raise ScheduleError("schedule must be a Schedule")
    tasks = {task.id: task for task in schedule.tasks}
    models = {model.id: model for model in schedule.models}
    conditions = {condition.id: condition for condition in schedule.conditions}
    expected_ids: list[str] = []
    seen_assignments: set[str] = set()
    seen_sessions: set[str] = set()
    seen_pairs: dict[str, list[Assignment]] = {}
    for index, row in enumerate(schedule.assignments):
        if row.order != index or row.index != index + 1 or row.schedule_id != schedule.schedule_id:
            raise ScheduleError("schedule row order or schedule identity is inconsistent")
        if row.task_id not in tasks or row.task_hash != tasks[row.task_id].task_hash:
            raise ScheduleError(f"task hash drift in row {index}")
        if row.model_id not in models:
            raise ScheduleError(f"unknown model in row {index}")
        model = models[row.model_id]
        if (row.provider, row.wire_model_id, row.reasoning, dict(row.reasoning_settings)) != (
            model.provider,
            model.wire_model_id,
            model.reasoning,
            dict(model.settings),
        ):
            raise ScheduleError(f"model configuration drift in row {index}")
        if row.condition not in conditions or row.condition_hash != conditions[row.condition].hash:
            raise ScheduleError(f"condition hash drift in row {index}")
        if row.repeat not in schedule.repeats or row.mode not in schedule.modes:
            raise ScheduleError(f"factorial value drift in row {index}")
        expected_assignment = assignment_id(
            task_id=row.task_id,
            task_hash=row.task_hash,
            model_id=row.model_id,
            provider=row.provider,
            wire_model_id=row.wire_model_id,
            reasoning=row.reasoning,
            condition=row.condition,
            mode=row.mode,
            repeat=row.repeat,
            run_id=schedule.run_id,
        )
        expected_pair = pair_id(
            task_id=row.task_id,
            task_hash=row.task_hash,
            model_id=row.model_id,
            provider=row.provider,
            wire_model_id=row.wire_model_id,
            reasoning=row.reasoning,
            condition=row.condition,
            repeat=row.repeat,
            run_id=schedule.run_id,
        )
        expected_session = session_id(
            task_id=row.task_id,
            task_hash=row.task_hash,
            model_id=row.model_id,
            provider=row.provider,
            wire_model_id=row.wire_model_id,
            reasoning=row.reasoning,
            condition=row.condition,
            mode=row.mode,
            repeat=row.repeat,
            run_id=schedule.run_id,
        )
        if (row.assignment_id, row.pair_id, row.session_id) != (
            expected_assignment,
            expected_pair,
            expected_session,
        ):
            raise ScheduleError(f"record identity drift in row {index}")
        if row.assignment_id in seen_assignments or row.session_id in seen_sessions:
            raise ScheduleError("assignment or session IDs are not unique")
        seen_assignments.add(row.assignment_id)
        seen_sessions.add(row.session_id)
        expected_ids.append(row.assignment_id)
        seen_pairs.setdefault(row.pair_id, []).append(row)
    expected_schedule = record_schedule_id(
        schedule.master_seed,
        expected_ids,
        algorithm=schedule.algorithm,
        run_id=schedule.run_id,
    )
    if expected_schedule != schedule.schedule_id:
        raise ScheduleError("schedule_id does not match ordered assignments")
    expected_count = (
        len(schedule.tasks)
        * len(schedule.models)
        * len(schedule.conditions)
        * len(schedule.modes)
        * len(schedule.repeats)
    )
    if len(schedule.assignments) != expected_count:
        raise ScheduleError("schedule row count is not factorial")
    for pair_rows in seen_pairs.values():
        if len(pair_rows) != len(schedule.modes) or {row.mode for row in pair_rows} != set(schedule.modes):
            raise ScheduleError("pair does not contain exactly one row per mode")
    for start in range(0, len(schedule.assignments), schedule.cell_count):
        block = schedule.assignments[start : start + schedule.cell_count]
        if len(block) != schedule.cell_count or {row.cell_index for row in block} != set(range(schedule.cell_count)):
            raise ScheduleError("round-robin block does not contain every model-repeat cell")
    return True


def load_schedule(path: str | Path) -> Schedule:
    """Load a canonical schedule manifest and rederive all identities."""

    target = Path(path)
    try:
        value = strict_loads(target.read_bytes())
    except (OSError, CanonicalJSONError, CanonicalizationError, ValueError) as exc:
        raise ScheduleError(f"invalid schedule manifest: {target}") from exc
    if not isinstance(value, Mapping) or value.get("schema_version") != SCHEMA_VERSION:
        raise ScheduleError("schedule manifest has the wrong schema version")
    try:
        schedule = build_schedule(
            value.get("tasks"),
            value.get("models"),
            conditions=value.get("conditions"),
            modes=value.get("modes"),
            repeats=value.get("repeats"),
            master_seed=value.get("master_seed"),
            algorithm=value.get("algorithm"),
            run_id=value.get("run_id", ""),
        )
    except (ScheduleError, TypeError, ValueError) as exc:
        raise ScheduleError("schedule manifest failed identity validation") from exc
    if schedule.to_manifest() != value:
        raise ScheduleError("schedule manifest is not the canonical generated manifest")
    return schedule


# Compatibility aliases used by early smoke scripts and notebooks.
generate_schedule = build_schedule
make_schedule = build_schedule
create_schedule = build_schedule
schedule_assignments = build_schedule


__all__ = [
    "Assignment",
    "DEFAULT_MASTER_SEED",
    "DEFAULT_MODEL_CONFIGS",
    "DEFAULT_MODES",
    "DEFAULT_REPEATS",
    "MODE_ORDER",
    "ModelConfig",
    "SCHEDULE_ALGORITHM",
    "SCHEMA_VERSION",
    "Schedule",
    "ScheduleError",
    "TaskRef",
    "build_pilot_schedule",
    "build_schedule",
    "create_schedule",
    "generate_schedule",
    "make_schedule",
    "normalize_modes",
    "normalize_models",
    "normalize_repeats",
    "normalize_tasks",
    "load_schedule",
    "schedule_rank",
    "schedule_assignments",
    "validate_schedule",
    "write_schedule",
]
