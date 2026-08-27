"""Offline verification of the trusted Lean task corpus.

This module is the evidence boundary between the checked-in corpus and the
later evaluator.  It compiles the six proof sidecars and the two independent
negation witnesses with the pinned Lean binary.  It also records the two
``#eval decide`` results and a fresh ``#print axioms`` result for every
compiled source.

The verifier is intentionally independent of the model protocol and of the
bounded candidate runner.  It never calls a provider, never writes into the
repository, and uses a fresh temporary directory for each Lean process.
Report bytes are canonical and contain no elapsed times or temporary paths,
so repeating a successful precheck produces the same report.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

from . import corpus
from .canonical import canonical_bytes, canonical_dumps, domain_hash


REPORT_SCHEMA_VERSION = "lean-eval/corpus-precheck-v1"
REPORT_HASH_DOMAIN = "lean-eval/source"
DEFAULT_TIMEOUT_SECONDS = 5.0
DEFAULT_MAX_OUTPUT_BYTES = 32 * 1024
DEFAULT_MAX_SOURCE_BYTES = 1 * 1024 * 1024
PROBE_AXIOM_NAME = "__rh_witness_axiom_probe"
LEAN_FLAGS: tuple[str, ...] = (
    "--json",
    "--quiet",
    "--trust=0",
    "-j",
    "1",
    "-M",
    "512",
    "-T",
    "2000000",
)


class CorpusVerificationError(RuntimeError):
    """Raised by the strict verification entry point."""


@dataclass(frozen=True)
class ProcessResult:
    """Small, non-persistent result from one Lean child process."""

    returncode: int | None
    signal: int | None
    stdout: bytes
    stderr: bytes
    limit: str | None
    elapsed_ms: int


@dataclass(frozen=True)
class AxiomReport:
    """Parsed, target-bound output from one ``#print axioms`` command."""

    target: str
    observed: tuple[str, ...]
    allowed: tuple[str, ...]
    delta: tuple[str, ...]
    status: str
    count: int
    diagnostic_sha256: str
    exit_code: int | None = None
    signal: int | None = None
    limit: str | None = None
    error_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "observed": list(self.observed),
            "allowed": list(self.allowed),
            "delta": list(self.delta),
            "status": self.status,
            "count": self.count,
            "counts": {"reports": self.count, "errors": self.error_count},
            "diagnostic_sha256": self.diagnostic_sha256,
            "exit_code": self.exit_code,
            "signal": self.signal,
            "limit": self.limit,
            "error_count": self.error_count,
        }


@dataclass(frozen=True)
class ProbeRecord:
    """Deterministic facts from one oracle or witness compilation."""

    task_id: str
    task_hash: str
    kind: str
    sidecar: str
    sidecar_sha256: str
    template_sha256: str
    fixed_region_sha256: str
    source_sha256: str
    probe_source_sha256: str
    axiom_target: str
    expected_axioms: tuple[str, ...]
    exit_code: int | None
    signal: int | None
    limit: str | None
    compiler_ok: bool
    axioms: tuple[str, ...]
    axiom_status: str
    axiom_delta: tuple[str, ...]
    axiom_report_count: int
    axiom_diagnostic_sha256: str
    axiom_report: AxiomReport
    evaluator_outputs: tuple[str, ...]
    evaluator_false: bool | None
    status: str
    reason_codes: tuple[str, ...]
    stdout: str
    stderr: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible record with stable field types."""

        return {
            "task_id": self.task_id,
            "task_hash": self.task_hash,
            "kind": self.kind,
            "sidecar": self.sidecar,
            "sidecar_sha256": self.sidecar_sha256,
            "template_sha256": self.template_sha256,
            "fixed_region_sha256": self.fixed_region_sha256,
            "source_sha256": self.source_sha256,
            "probe_source_sha256": self.probe_source_sha256,
            "axiom_target": self.axiom_target,
            "expected_axioms": list(self.expected_axioms),
            "exit_code": self.exit_code,
            "signal": self.signal,
            "limit": self.limit,
            "compiler_ok": self.compiler_ok,
            "axioms": list(self.axioms),
            "axiom_status": self.axiom_status,
            "axiom_delta": list(self.axiom_delta),
            "axiom_report_count": self.axiom_report_count,
            "axiom_diagnostic_sha256": self.axiom_diagnostic_sha256,
            "axiom_report": self.axiom_report.to_dict(),
            "evaluator_outputs": list(self.evaluator_outputs),
            "evaluator_false": self.evaluator_false,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


def _as_path(value: str | Path | None, *, default: Path) -> Path:
    return (Path(value) if value is not None else default).resolve()


def _project_root(root: str | Path | None) -> Path:
    return _as_path(root, default=corpus.PROJECT_ROOT)


def _manifest_path(root: Path, value: str | Path | None) -> Path:
    return _as_path(value, default=root / "lean_eval" / "tasks" / "manifest.json")


def _tasks_root(root: Path) -> Path:
    return root / "lean_eval" / "tasks"


def _strict_regular(path: Path, *, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise CorpusVerificationError(f"{label} is not a regular file: {path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise CorpusVerificationError(f"cannot read {label}: {path}") from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _source_hash(payload: bytes) -> str:
    return corpus.hash_source(payload)


def _sanitize_text(value: bytes | str, *, temp_roots: Iterable[str] = ()) -> str:
    """Decode compiler output and remove machine-specific workspace paths."""

    if isinstance(value, bytes):
        text = value.decode("utf-8", "replace")
    else:
        text = value
    result: list[str] = []
    index = 0
    while index < len(text):
        if text[index] == "\x1b":
            index += 1
            if index < len(text) and text[index] == "[":
                index += 1
                while index < len(text) and not ("@" <= text[index] <= "~"):
                    index += 1
                if index < len(text):
                    index += 1
            else:
                index += 1
            continue
        result.append(text[index])
        index += 1
    text = "".join(result)
    for root in temp_roots:
        if root:
            text = text.replace(root, "<workspace>")
    text = re.sub(
        r"/(?:private/)?tmp/lean-eval-corpus-[^/\\\"'[:space:]]+",
        "<workspace>",
        text,
    )
    return text


def _stable_error(error: BaseException) -> str:
    """Keep failure reports useful while removing host-specific path data."""

    message = str(error)
    message = message.replace(str(corpus.PROJECT_ROOT), "<project>")
    message = re.sub(r"/(?:private/)?tmp/[^/\\\"'[:space:]]+", "<tmp>", message)
    return message


def _minimal_environment(home: Path, temporary: Path) -> dict[str, str]:
    """Construct an allowlisted environment for a trusted probe."""

    return {
        "HOME": str(home),
        "TMPDIR": str(temporary),
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    """Kill the entire child process group when the platform supports it."""

    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:  # pragma: no cover - the pinned development host is POSIX
            process.kill()
    except (OSError, ProcessLookupError):
        try:
            process.kill()
        except (OSError, ProcessLookupError):
            pass


def _run_lean(
    source: bytes,
    *,
    binary: Path,
    timeout_seconds: float,
    max_output_bytes: int,
) -> ProcessResult:
    """Run one source in a fresh read-only workspace with bounded output."""

    if len(source) > DEFAULT_MAX_SOURCE_BYTES:
        return ProcessResult(None, None, b"", b"", "source_limit", 0)
    if timeout_seconds <= 0 or max_output_bytes <= 0:
        raise ValueError("timeout and output limits must be positive")

    temporary_parent = Path("/private/tmp") if Path("/private/tmp").is_dir() else None
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="lean-eval-corpus-", dir=temporary_parent) as name:
        workspace = Path(name)
        home = workspace / "home"
        temporary = workspace / "tmp"
        home.mkdir(mode=0o700)
        temporary.mkdir(mode=0o700)
        source_path = workspace / "Main.lean"
        source_path.write_bytes(source)
        os.chmod(source_path, 0o400)
        command = [str(binary), *LEAN_FLAGS, "Main.lean"]
        try:
            process = subprocess.Popen(
                command,
                cwd=workspace,
                env=_minimal_environment(home, temporary),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                close_fds=True,
                pass_fds=(),
                start_new_session=True,
            )
        except OSError as exc:
            elapsed = int((time.monotonic() - started) * 1000)
            return ProcessResult(
                None,
                None,
                b"",
                str(exc).encode("utf-8", "replace"),
                "runner_error",
                elapsed,
            )

        assert process.stdout is not None
        assert process.stderr is not None
        selector = selectors.DefaultSelector()
        for stream in (process.stdout, process.stderr):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ)
        output = {process.stdout: bytearray(), process.stderr: bytearray()}
        limit: str | None = None
        deadline = started + timeout_seconds

        while selector.get_map():
            now = time.monotonic()
            if limit is None and now >= deadline:
                limit = "timeout"
                _terminate_process(process)
            events = selector.select(0.05)
            for key, _ in events:
                stream = key.fileobj
                try:
                    chunk = os.read(stream.fileno(), 4096)
                except BlockingIOError:
                    continue
                except OSError:
                    chunk = b""
                if not chunk:
                    try:
                        selector.unregister(stream)
                    except Exception:
                        pass
                    continue
                remaining = max_output_bytes - sum(len(data) for data in output.values())
                if remaining <= 0:
                    limit = limit or "output_limit"
                    _terminate_process(process)
                    break
                if len(chunk) > remaining:
                    output[stream].extend(chunk[:remaining])
                    limit = limit or "output_limit"
                    _terminate_process(process)
                    break
                output[stream].extend(chunk)
            if limit is not None:
                break
            if process.poll() is not None and not events:
                continue

        if limit is not None:
            for stream in (process.stdout, process.stderr):
                try:
                    selector.unregister(stream)
                except Exception:
                    pass
        selector.close()
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            _terminate_process(process)
            process.wait(timeout=1.0)
        for stream in (process.stdout, process.stderr):
            try:
                stream.close()
            except OSError:
                pass
        returncode = process.returncode
        child_signal = -returncode if returncode is not None and returncode < 0 else None
        elapsed = int((time.monotonic() - started) * 1000)
        return ProcessResult(
            returncode,
            child_signal,
            bytes(output[process.stdout]),
            bytes(output[process.stderr]),
            limit,
            elapsed,
        )


def _version_preflight(binary: Path, expected_sha256: str) -> dict[str, Any]:
    """Check the direct executable and its exact Lean version identity."""

    if not binary.is_absolute():
        raise CorpusVerificationError("Lean binary must be an absolute path")
    if binary.is_symlink() or not binary.is_file() or not os.access(binary, os.X_OK):
        raise CorpusVerificationError(f"pinned Lean binary is not executable: {binary}")
    actual_sha256 = corpus.sha256_file(binary)
    if actual_sha256 != expected_sha256:
        raise CorpusVerificationError(
            f"pinned Lean binary digest mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    version_parent = Path("/private/tmp") if Path("/private/tmp").is_dir() else None
    with tempfile.TemporaryDirectory(prefix="lean-eval-version-", dir=version_parent) as name:
        workspace = Path(name)
        home = workspace / "home"
        temporary = workspace / "tmp"
        home.mkdir(mode=0o700)
        temporary.mkdir(mode=0o700)
        try:
            result = subprocess.run(
                [str(binary), "--version"],
                cwd=workspace,
                env=_minimal_environment(home, temporary),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                close_fds=True,
                timeout=5.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CorpusVerificationError(f"Lean version preflight failed: {exc}") from exc
    stdout = _sanitize_text(result.stdout).strip()
    stderr = _sanitize_text(result.stderr).strip()
    if result.returncode != 0:
        raise CorpusVerificationError(f"Lean --version exited {result.returncode}: {stderr or stdout}")
    if not stdout:
        raise CorpusVerificationError("Lean --version returned no output")
    if "version 4.30.0" not in stdout or corpus.LEAN_COMMIT not in stdout:
        raise CorpusVerificationError("Lean --version does not match the pinned version and commit")
    return {
        "binary": binary.name,
        "binary_sha256": actual_sha256,
        "version_output": stdout,
        "version_stderr": stderr,
        "exit_code": result.returncode,
    }


def parse_lean_json_events(output: bytes | str) -> tuple[dict[str, Any], ...]:
    """Parse JSON diagnostics emitted by Lean's ``--json`` mode."""

    text = output.decode("utf-8", "replace") if isinstance(output, bytes) else output
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            events.append(value)
    return tuple(events)


def evaluator_outputs(output: bytes | str) -> tuple[str, ...]:
    """Return exact boolean values emitted by trusted ``#eval`` commands."""

    events = parse_lean_json_events(output)
    values: list[str] = []
    for event in events:
        data = event.get("data")
        severity = event.get("severity")
        if severity not in (None, "information") or not isinstance(data, str):
            continue
        value = data.strip()
        if value in {"true", "false"}:
            values.append(value)
    return tuple(values)


_AXIOM_CLEAN_RE = re.compile(r"^'(?P<target>[^']+)' does not depend on any axioms$")
_AXIOM_DEPEND_RE = re.compile(
    r"^'(?P<target>[^']+)' depends on axioms: \[(?P<axioms>.*)\]$"
)
_AXIOM_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_'.]*$")


def _axiom_messages(output: bytes | str) -> list[tuple[str, tuple[str, ...] | None]]:
    """Return strict target/axiom pairs from compiler information messages."""

    text = output.decode("utf-8", "replace") if isinstance(output, bytes) else output
    events = parse_lean_json_events(text)
    messages: list[str] = []
    for event in events:
        data = event.get("data")
        severity = event.get("severity")
        if isinstance(data, str) and severity in (None, "information"):
            messages.append(data.strip())
    if not messages:
        messages = [line.strip() for line in text.splitlines()]
    reports: list[tuple[str, tuple[str, ...] | None]] = []
    for message in messages:
        clean = _AXIOM_CLEAN_RE.fullmatch(message)
        if clean is not None:
            reports.append((clean.group("target"), ()))
            continue
        dependent = _AXIOM_DEPEND_RE.fullmatch(message)
        if dependent is None:
            continue
        raw_names = dependent.group("axioms").strip()
        if not raw_names:
            reports.append((dependent.group("target"), ()))
            continue
        names = tuple(name.strip() for name in raw_names.split(","))
        if any(not _AXIOM_NAME_RE.fullmatch(name) for name in names):
            reports.append((dependent.group("target"), None))
            continue
        if len(set(names)) != len(names):
            reports.append((dependent.group("target"), None))
            continue
        reports.append((dependent.group("target"), names))
    return reports


def _has_error_diagnostics(output: bytes | str) -> bool:
    text = output.decode("utf-8", "replace") if isinstance(output, bytes) else output
    for event in parse_lean_json_events(text):
        if event.get("severity") in {"error", "fatal"}:
            return True
    return False


def parse_axioms(output: bytes | str, expected_target: str | None = None) -> tuple[str, ...] | None:
    """Extract one fresh ``#print axioms`` result from Lean output."""

    reports = _axiom_messages(output)
    if expected_target is not None and len(reports) == 1 and reports[0][0] != expected_target:
        return None
    if len(reports) != 1:
        return None
    return reports[0][1]


def parse_axiom_report(
    output: bytes | str,
    *,
    target: str,
    allowed: Sequence[str] = (),
) -> AxiomReport:
    """Build a target-bound axiom report from one Lean output stream."""

    all_reports = _axiom_messages(output)
    reports = [report for report in all_reports if report[0] == target]
    observed = reports[0][1] if len(all_reports) == 1 and reports[0][1] is not None else ()
    allowed_tuple = tuple(allowed)
    delta = tuple(sorted(set(observed).difference(allowed_tuple)))
    if len(all_reports) != 1:
        status = "missing" if not all_reports else "malformed_or_duplicate"
    elif not reports:
        status = "wrong_target"
    elif reports[0][1] is None:
        status = "malformed"
    elif any(name in {"sorryAx", "admitAx"} for name in observed):
        status = "forbidden"
    elif set(observed) == set(allowed_tuple):
        status = "clean"
    else:
        status = "unexpected"
    diagnostic = _sanitize_text(output).encode("utf-8", "replace")
    return AxiomReport(
        target=target,
        observed=tuple(observed),
        allowed=allowed_tuple,
        delta=delta,
        status=status,
        count=len(reports),
        diagnostic_sha256=_sha256(diagnostic),
    )


def _qualified(namespace: str, name: str) -> str:
    return f"{namespace}.{name}"


def _witness_probe_source(raw: bytes) -> tuple[bytes, str]:
    """Name the anonymous negation witness in a temporary probe copy."""

    marker = b"example :"
    if raw.count(marker) != 1:
        raise CorpusVerificationError("witness must contain exactly one anonymous example")
    renamed = raw.replace(marker, f"theorem {PROBE_AXIOM_NAME} :".encode("ascii"), 1)
    target = _qualified(corpus.NAMESPACE, PROBE_AXIOM_NAME)
    return renamed + b"\n#print axioms " + target.encode("ascii") + b"\n", target


def _oracle_probe_source(materialized: bytes, theorem: str) -> tuple[bytes, str]:
    target = _qualified(corpus.NAMESPACE, theorem)
    return materialized + b"\n#print axioms " + target.encode("ascii") + b"\n", target


def _parenthesized_oracle_source(
    task: corpus.TaskSpec,
    proof: bytes,
    *,
    root: Path,
) -> bytes:
    """Insert an oracle as one parenthesized proof term in the trusted file."""

    prefix, suffix = corpus.template_parts(task, root=root)
    return prefix + b"(" + proof + b")" + suffix


def _probe_record(
    task: corpus.TaskSpec,
    *,
    kind: str,
    sidecar: str,
    sidecar_sha256: str,
    materialized: bytes,
    probe_source: bytes,
    axiom_target: str,
    binary: Path,
    timeout_seconds: float,
    max_output_bytes: int,
) -> ProbeRecord:
    if len(probe_source) > DEFAULT_MAX_SOURCE_BYTES:
        process = ProcessResult(None, None, b"", b"", "source_limit", 0)
    else:
        process = _run_lean(
            probe_source,
            binary=binary,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )
    combined = process.stdout + process.stderr
    sanitized_stdout = _sanitize_text(process.stdout)
    sanitized_stderr = _sanitize_text(process.stderr)
    axiom_report = parse_axiom_report(
        combined,
        target=axiom_target,
        allowed=task.allowed_axioms,
    )
    axiom_report = replace(
        axiom_report,
        exit_code=process.returncode,
        signal=process.signal,
        limit=process.limit,
        error_count=sum(
            event.get("severity") in {"error", "fatal"}
            for event in parse_lean_json_events(combined)
        ),
    )
    parsed_axioms = axiom_report.observed
    axiom_status = axiom_report.status
    outputs = evaluator_outputs(process.stdout)
    false_result: bool | None
    if kind == "witness":
        false_result = outputs == ("false",)
    else:
        false_result = None
    reasons: list[str] = []
    if process.limit is not None:
        reasons.append(process.limit)
    if process.returncode != 0:
        reasons.append("lean_nonzero" if process.limit is None else "lean_after_limit")
    if _has_error_diagnostics(combined):
        reasons.append("lean_error_diagnostic")
    if axiom_status != "clean":
        reasons.append(
            "axiom_report_missing"
            if axiom_status == "missing"
            else "axiom_report_malformed"
            if axiom_status in {"malformed", "malformed_or_duplicate"}
            else "axiom_forbidden"
            if axiom_status == "forbidden"
            else "axiom_report_wrong_target"
            if axiom_status == "wrong_target"
            else "axiom_delta"
        )
    if kind == "witness" and false_result is not True:
        reasons.append("evaluator_not_false")
    if kind == "oracle" and outputs:
        reasons.append("unexpected_evaluator_output")
    compiler_ok = process.returncode == 0 and process.limit is None and not _has_error_diagnostics(combined)
    status = "pass" if compiler_ok and not reasons else "fail"
    return ProbeRecord(
        task_id=task.id,
        task_hash=task.task_hash,
        kind=kind,
        sidecar=sidecar,
        sidecar_sha256=sidecar_sha256,
        template_sha256=task.template_sha256,
        fixed_region_sha256=task.fixed_region_sha256,
        source_sha256=_source_hash(materialized),
        probe_source_sha256=_source_hash(probe_source),
        axiom_target=axiom_target,
        expected_axioms=tuple(task.allowed_axioms),
        exit_code=process.returncode,
        signal=process.signal,
        limit=process.limit,
        compiler_ok=compiler_ok,
        axioms=parsed_axioms,
        axiom_status=axiom_status,
        axiom_delta=axiom_report.delta,
        axiom_report_count=axiom_report.count,
        axiom_diagnostic_sha256=axiom_report.diagnostic_sha256,
        axiom_report=axiom_report,
        evaluator_outputs=outputs,
        evaluator_false=false_result,
        status=status,
        reason_codes=tuple(sorted(set(reasons))),
        stdout=sanitized_stdout,
        stderr=sanitized_stderr,
    )


def _manifest_identity(manifest_raw: bytes, manifest: Mapping[str, Any]) -> dict[str, Any]:
    task_hashes: dict[str, str] = {}
    source_hashes: dict[str, Any] = {}
    for item in manifest["tasks"]:
        task_hashes[item["id"]] = item["task_hash"]
        source_hashes[item["id"]] = {
            "template_sha256": item["template_sha256"],
            "fixed_region_sha256": item["fixed_region_sha256"],
            "sidecar_sha256": item.get("oracle_sha256") or item.get("negative_witness_sha256"),
        }
    return {
        "path": "lean_eval/tasks/manifest.json",
        "sha256": _sha256(manifest_raw),
        "canonical_hash": corpus.manifest_hash(manifest),
        "task_hashes": task_hashes,
        "source_hashes": source_hashes,
    }


def _empty_report(
    *,
    manifest_identity: Mapping[str, Any] | None = None,
    errors: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "deterministic": True,
        "status": "fail",
        "manifest": dict(manifest_identity or {}),
        "toolchain": {},
        "probes": [],
        "summary": {
            "task_count": 0,
            "oracle_count": 0,
            "witness_count": 0,
            "oracle_exit_zero": 0,
            "witness_exit_zero": 0,
            "false_evaluator_count": 0,
            "clean_axiom_count": 0,
            "source_validation": False,
            "toolchain_preflight": False,
            "all_allowed_axioms": False,
            "all_false_evaluators": False,
        },
        "errors": list(errors),
    }


def _report_body(report: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(report)
    body.pop("report_sha256", None)
    return body


def report_sha256(report: Mapping[str, Any]) -> str:
    """Hash a report body while excluding its optional self-digest field."""

    return domain_hash(REPORT_HASH_DOMAIN, canonical_bytes(_report_body(report)))


def report_bytes(report: Mapping[str, Any]) -> bytes:
    """Serialize a report deterministically with one final LF."""

    checked = dict(report)
    checked["report_sha256"] = report_sha256(checked)
    return canonical_bytes(checked) + b"\n"


def verify_corpus(
    *,
    root: str | Path | None = None,
    manifest_path: str | Path | None = None,
    lean_binary: str | Path | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> dict[str, Any]:
    """Compile every trusted sidecar and return a deterministic report.

    Validation and binary preflight happen before the first Lean invocation.
    Failures are represented as a report so a command-line caller can archive
    the exact machine-readable reason.  Use :func:`verify_corpus_or_raise`
    when a Python caller wants exception semantics.
    """

    project_root = _project_root(root)
    manifest_file = _manifest_path(project_root, manifest_path)
    default_binary = corpus.DEFAULT_LEAN_BINARY if lean_binary is None else Path(lean_binary)
    manifest_identity: dict[str, Any] | None = None
    try:
        manifest_raw = _strict_regular(manifest_file, label="manifest")
        manifest = corpus.parse_manifest_bytes(manifest_raw)
        if canonical_bytes(manifest) + b"\n" != manifest_raw:
            raise CorpusVerificationError("manifest bytes are not canonical")
        manifest_identity = _manifest_identity(manifest_raw, manifest)
        tasks = corpus.validate_sources(manifest, root=project_root)
    except Exception as exc:
        return _empty_report(manifest_identity=manifest_identity, errors=(_stable_error(exc),))

    report = _empty_report(manifest_identity=manifest_identity)
    report["summary"]["task_count"] = len(tasks)
    report["summary"]["source_validation"] = True
    try:
        expected_binary = manifest["lean"]["absolute_binary_sha256"]
        toolchain = _version_preflight(Path(default_binary), expected_binary)
    except Exception as exc:
        report["errors"] = [_stable_error(exc)]
        return report
    report["toolchain"] = {
        "manifest_toolchain": manifest["lean"]["toolchain"],
        "manifest_commit": manifest["lean"]["commit"],
        **toolchain,
    }
    report["summary"]["toolchain_preflight"] = True

    records: list[ProbeRecord] = []
    tasks_root = _tasks_root(project_root)
    try:
        for task in tasks:
            if task.oracle is not None:
                sidecar_path = tasks_root / task.oracle
                sidecar_raw = _strict_regular(sidecar_path, label=f"oracle {task.id}")
                materialized = _parenthesized_oracle_source(task, sidecar_raw, root=project_root)
                probe_source, target = _oracle_probe_source(materialized, task.theorem)
                records.append(
                    _probe_record(
                        task,
                        kind="oracle",
                        sidecar=task.oracle,
                        sidecar_sha256=task.oracle_sha256 or "",
                        materialized=materialized,
                        probe_source=probe_source,
                        axiom_target=target,
                        binary=Path(default_binary),
                        timeout_seconds=timeout_seconds,
                        max_output_bytes=max_output_bytes,
                    )
                )
            elif task.negative_witness is not None:
                sidecar_path = tasks_root / task.negative_witness
                sidecar_raw = _strict_regular(sidecar_path, label=f"witness {task.id}")
                probe_source, target = _witness_probe_source(sidecar_raw)
                records.append(
                    _probe_record(
                        task,
                        kind="witness",
                        sidecar=task.negative_witness,
                        sidecar_sha256=task.negative_witness_sha256 or "",
                        materialized=sidecar_raw,
                        probe_source=probe_source,
                        axiom_target=target,
                        binary=Path(default_binary),
                        timeout_seconds=timeout_seconds,
                        max_output_bytes=max_output_bytes,
                    )
                )
            else:  # pragma: no cover - corpus validation enforces this partition
                report["errors"].append(f"task has no sidecar: {task.id}")
    except Exception as exc:
        report["errors"] = [_stable_error(exc)]

    report["probes"] = [record.to_dict() for record in records]
    oracle_records = [record for record in records if record.kind == "oracle"]
    witness_records = [record for record in records if record.kind == "witness"]
    clean_axioms = sum(record.axiom_status == "clean" for record in records)
    false_evaluators = sum(record.evaluator_false is True for record in witness_records)
    report["summary"].update(
        {
            "oracle_count": len(oracle_records),
            "witness_count": len(witness_records),
            "oracle_exit_zero": sum(record.compiler_ok for record in oracle_records),
            "witness_exit_zero": sum(record.compiler_ok for record in witness_records),
            "false_evaluator_count": false_evaluators,
            "clean_axiom_count": clean_axioms,
            "all_allowed_axioms": clean_axioms == len(records),
            "all_false_evaluators": false_evaluators == 2 and len(witness_records) == 2,
        }
    )
    report["status"] = (
        "pass"
        if len(records) == 8
        and all(record.status == "pass" for record in records)
        and report["summary"]["all_false_evaluators"]
        else "fail"
    )
    return report


def verify_corpus_or_raise(**kwargs: Any) -> dict[str, Any]:
    """Run :func:`verify_corpus` and raise if the report is not a pass."""

    report = verify_corpus(**kwargs)
    if report.get("status") != "pass":
        raise CorpusVerificationError(canonical_dumps(report))
    return report


run_precheck = verify_corpus
precheck = verify_corpus
compile_corpus = verify_corpus


def write_report(report: Mapping[str, Any], path: str | Path) -> Path:
    """Write a canonical report to an explicitly selected artifact path."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(report_bytes(report))
    return target


def _self_test() -> int:
    report = verify_corpus()
    encoded = report_bytes(report)
    decoded = corpus.strict_loads(encoded)
    if report.get("status") != "pass" or not isinstance(decoded, dict):
        print(canonical_dumps(report))
        return 1
    if encoded != report_bytes(decoded):
        print(canonical_dumps({"status": "fail", "error": "report is not deterministic"}))
        return 1
    print(canonical_dumps({"status": "pass", "report_sha256": report_sha256(report)}))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="verify the trusted Lean evaluation corpus")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--lean-binary", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-output", type=int, default=DEFAULT_MAX_OUTPUT_BYTES)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--precheck", action="store_true", help="run the corpus precheck")
    args = parser.parse_args(argv)
    if args.self_test:
        return _self_test()
    report = verify_corpus(
        root=args.root,
        manifest_path=args.manifest,
        lean_binary=args.lean_binary,
        timeout_seconds=args.timeout,
        max_output_bytes=args.max_output,
    )
    encoded = report_bytes(report)
    if args.output is not None:
        write_report(report, args.output)
    else:
        print(encoded.decode("utf-8"), end="")
    return 0 if report.get("status") == "pass" else 1


if __name__ == "__main__":  # pragma: no cover - exercised by command checks
    raise SystemExit(main())


__all__ = [
    "AxiomReport",
    "CorpusVerificationError",
    "DEFAULT_MAX_OUTPUT_BYTES",
    "DEFAULT_MAX_SOURCE_BYTES",
    "DEFAULT_TIMEOUT_SECONDS",
    "LEAN_FLAGS",
    "PROBE_AXIOM_NAME",
    "ProbeRecord",
    "ProcessResult",
    "REPORT_SCHEMA_VERSION",
    "evaluator_outputs",
    "main",
    "parse_axioms",
    "parse_axiom_report",
    "parse_lean_json_events",
    "precheck",
    "report_bytes",
    "report_sha256",
    "run_precheck",
    "verify_corpus",
    "verify_corpus_or_raise",
    "write_report",
]
