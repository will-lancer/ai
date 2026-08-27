"""Guardrails for the local part of the lean-reward-hacking workflow.

The experiment's full runs belong in Colab.  This module gives local tests,
inspection, and compact analysis a small, explicit execution boundary.  It
does not try to turn the Mac into a batch scheduler: an operation that looks
like training, a sweep, a download, or accelerator work is rejected before a
child process starts.

The memory limit has two layers.  Requested budgets are checked portably, and
POSIX children receive ``RLIMIT_AS``/``RLIMIT_CPU`` where the platform exposes
them.  macOS does not promise a hard per-process RSS ceiling from
``RLIMIT_AS``.  Callers should treat the 4 GB check as a contract guard and
keep the actual local jobs small; Colab remains the execution target for
large work.
"""

from __future__ import annotations

import contextlib
import math
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence


THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "GOMAXPROCS",
)

LOCAL_OPERATION_KINDS: frozenset[str] = frozenset(
    {"inspect", "edit", "test", "smoke", "analysis", "plot", "verify"}
)
REMOTE_OPERATION_KINDS: frozenset[str] = frozenset(
    {"training", "train", "sweep", "basin", "gpu", "download", "inference", "fine_tune"}
)


class ContractViolation(RuntimeError):
    """Raised when a local operation would exceed the project contract."""


@dataclass(frozen=True)
class ComputeBudget:
    """Maximum resources a local child may request.

    Values above the project ceiling are rejected at construction time.  A
    smaller budget is useful for focused tests, including a short timeout in
    a timeout unit test.
    """

    max_cores: int = 2
    max_ram_gb: float = 4.0
    max_seconds: float = 300.0
    allow_gpu: bool = False

    def __post_init__(self) -> None:
        if self.max_cores < 1 or self.max_cores > 2:
            raise ContractViolation("local CPU budget must be between 1 and 2 cores")
        if not math.isfinite(self.max_ram_gb) or self.max_ram_gb <= 0 or self.max_ram_gb > 4.0:
            raise ContractViolation("local memory budget must be in (0, 4] GB")
        if not math.isfinite(self.max_seconds) or self.max_seconds <= 0 or self.max_seconds > 300.0:
            raise ContractViolation("local wall-time budget must be in (0, 300] seconds")
        if self.allow_gpu:
            raise ContractViolation("local GPU execution is outside the project contract")

    @property
    def max_ram_bytes(self) -> int:
        return int(self.max_ram_gb * (1024**3))


@dataclass(frozen=True)
class GuardedCommandResult:
    """Small, serialisable summary of a guarded subprocess."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    requested_memory_gb: float
    operation_kind: str


def _normalise_kind(kind: str | None) -> str | None:
    if kind is None:
        return None
    normalised = kind.strip().lower().replace("-", "_")
    aliases = {"fine_tuning": "fine_tune", "finetune": "fine_tune", "tests": "test"}
    return aliases.get(normalised, normalised)


def _command_text(argv: Sequence[str]) -> str:
    return " ".join(str(part).lower() for part in argv)


def infer_operation_kind(argv: Sequence[str]) -> str:
    """Infer a conservative kind for a command whose caller omitted one."""

    text = _command_text(argv)
    # Explicit flags have priority over ordinary words in a module name.
    if any(flag in text for flag in ("--gpu", "--cuda", "--mps", "--device=cuda", "--device=mps")):
        return "gpu"
    if any(flag in text for flag in ("--sweep", "--basin", "--full-run", "--full_run")):
        return "sweep"
    if any(word in text for word in ("download", "wget", "curl", "git clone", "pip install")):
        return "download"
    if any(word in text for word in ("fine_tune", "finetune", "fine-tuning", "training", " train ")):
        return "training"
    return "verify"


def validate_requested_resources(
    *,
    requested_cores: int = 2,
    requested_memory_gb: float = 4.0,
    requested_seconds: float = 300.0,
    use_gpu: bool = False,
    operation_kind: str | None = None,
    budget: ComputeBudget | None = None,
) -> ComputeBudget:
    """Validate a local resource request and return its effective budget.

    ``operation_kind`` is mandatory in spirit for training code.  Supplying a
    remote kind makes the rejection explicit even if the command text is
    innocuous.  The returned budget is always CPU-only.
    """

    effective = budget or ComputeBudget()
    kind = _normalise_kind(operation_kind)
    if kind in REMOTE_OPERATION_KINDS:
        raise ContractViolation(f"operation kind {kind!r} must run through Google Colab")
    if kind is not None and kind not in LOCAL_OPERATION_KINDS:
        raise ContractViolation(f"unknown operation kind {kind!r}; use an explicit local or Colab kind")
    if requested_cores < 1 or requested_cores > effective.max_cores or requested_cores > 2:
        raise ContractViolation("requested CPU count exceeds the local two-core ceiling")
    if not math.isfinite(requested_memory_gb) or requested_memory_gb <= 0:
        raise ContractViolation("requested memory must be a positive finite value")
    if requested_memory_gb > effective.max_ram_gb or requested_memory_gb > 4.0:
        raise ContractViolation("requested memory exceeds the local 4 GB ceiling")
    if not math.isfinite(requested_seconds) or requested_seconds <= 0:
        raise ContractViolation("requested wall time must be a positive finite value")
    if requested_seconds > effective.max_seconds or requested_seconds > 300.0:
        raise ContractViolation("requested wall time exceeds the local five-minute ceiling")
    if use_gpu or effective.allow_gpu:
        raise ContractViolation("local GPU, Metal, MPS, and CUDA execution is forbidden")
    return effective


def _gpu_environment_present(env: Mapping[str, str] | None = None) -> bool:
    values = os.environ if env is None else env
    cuda = values.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda and cuda != "-1":
        return True
    if values.get("NVIDIA_VISIBLE_DEVICES", "").strip() not in {"", "none", "void", "-1"}:
        return True
    if values.get("LRH_ACCELERATOR", "").strip().lower() in {"cuda", "mps", "metal", "gpu"}:
        return True
    # Avoid importing torch.  If a caller already imported it, honour its
    # availability result without making torch a package dependency.
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if bool(torch.cuda.is_available()):
                return True
            backends = getattr(torch, "backends", None)
            mps = getattr(backends, "mps", None)
            if mps is not None and bool(mps.is_available()):
                return True
        except Exception:
            # A partially imported optional module cannot establish that a GPU
            # is in use.  The environment checks above still apply.
            pass
    return False


def assert_cpu_only(env: Mapping[str, str] | None = None) -> None:
    """Raise if accelerator-related environment or loaded backends are active."""

    if _gpu_environment_present(env):
        raise ContractViolation("GPU/MPS/CUDA visibility detected for a local operation")


def is_colab_runtime(env: Mapping[str, str] | None = None) -> bool:
    """Return whether the caller supplied an explicit Colab runtime marker."""

    values = os.environ if env is None else env
    marker = values.get("LRH_RUNTIME", values.get("COLAB_RELEASE_TAG", ""))
    return marker.strip().lower() in {"colab", "google_colab", "google-colab"}


def assert_colab_execution(*, require_gpu: bool = False, env: Mapping[str, str] | None = None) -> None:
    """Require an explicit Colab marker for full runs.

    The marker is deliberately opt-in.  A local process cannot silently turn a
    training command into an allowed operation by setting a device flag.
    """

    values = os.environ if env is None else env
    if not is_colab_runtime(values):
        raise ContractViolation("full training and sweeps require LRH_RUNTIME=colab")
    if require_gpu and not _gpu_environment_present(values):
        raise ContractViolation("this Colab configuration requires a visible GPU")


def _thread_environment(base: Mapping[str, str] | None, cores: int) -> dict[str, str]:
    result = dict(os.environ if base is None else base)
    value = str(max(1, min(cores, 2)))
    for name in THREAD_ENV_VARS:
        result[name] = value
    # These make accidental parallel BLAS/OpenMP pools less likely.  They do
    # not claim to control arbitrary Python threads; local work stays small.
    result["LRH_LOCAL_CORES"] = value
    result["LRH_LOCAL_MEMORY_GB"] = "4"
    result["LRH_LOCAL_WALL_SECONDS"] = "300"
    return result


@contextlib.contextmanager
def local_context(*, cores: int = 2, budget: ComputeBudget | None = None) -> Iterator[dict[str, str]]:
    """Temporarily set conservative numerical-thread variables in-process."""

    effective = validate_requested_resources(requested_cores=cores, budget=budget, operation_kind="test")
    old = {name: os.environ.get(name) for name in THREAD_ENV_VARS}
    old.update(
        {
            "LRH_LOCAL_CORES": os.environ.get("LRH_LOCAL_CORES"),
            "LRH_LOCAL_MEMORY_GB": os.environ.get("LRH_LOCAL_MEMORY_GB"),
            "LRH_LOCAL_WALL_SECONDS": os.environ.get("LRH_LOCAL_WALL_SECONDS"),
        }
    )
    updated = _thread_environment(os.environ, effective.max_cores)
    try:
        assert_cpu_only(updated)
        os.environ.update({name: updated[name] for name in updated if name in THREAD_ENV_VARS or name.startswith("LRH_LOCAL_")})
        yield updated
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _resource_preexec(budget: ComputeBudget, requested_seconds: float, requested_memory_gb: float):
    """Return a POSIX pre-exec hook, or ``None`` on platforms without resource."""

    if os.name != "posix":
        return None
    try:
        import resource
    except ImportError:
        return None

    cpu_seconds = max(1, int(math.ceil(requested_seconds)))
    memory_bytes = int(requested_memory_gb * (1024**3))

    def apply_limits() -> None:
        # ``resource`` calls may fail on macOS for a particular limit.  The
        # parent still enforces the wall timeout and validated request.
        for limit, value in (
            (resource.RLIMIT_CPU, cpu_seconds),
            (resource.RLIMIT_AS, memory_bytes),
        ):
            try:
                current_soft, current_hard = resource.getrlimit(limit)
                hard = current_hard if current_hard != resource.RLIM_INFINITY else value
                resource.setrlimit(limit, (min(value, hard), min(value, hard)))
            except (OSError, ValueError):
                continue

    return apply_limits


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except OSError:
            pass
    else:
        process.terminate()
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
        else:
            process.kill()


def run_guarded(
    argv: Sequence[str | os.PathLike[str]],
    *,
    operation_kind: str | None = None,
    budget: ComputeBudget | None = None,
    requested_cores: int = 2,
    requested_memory_gb: float = 4.0,
    requested_seconds: float = 300.0,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = True,
) -> GuardedCommandResult:
    """Run one local argv under the contract, with no shell interpretation."""

    args = tuple(os.fspath(part) for part in argv)
    if not args:
        raise ContractViolation("guarded command cannot be empty")
    kind = _normalise_kind(operation_kind) or infer_operation_kind(args)
    inferred_kind = infer_operation_kind(args)
    if inferred_kind in REMOTE_OPERATION_KINDS and kind in LOCAL_OPERATION_KINDS:
        raise ContractViolation(
            f"command text indicates {inferred_kind!r}; full training, sweeps, downloads, and accelerators run in Colab"
        )
    effective = validate_requested_resources(
        requested_cores=requested_cores,
        requested_memory_gb=requested_memory_gb,
        requested_seconds=requested_seconds,
        operation_kind=kind,
        budget=budget,
    )
    merged_env = _thread_environment(env, min(requested_cores, effective.max_cores))
    assert_cpu_only(merged_env)
    if kind in REMOTE_OPERATION_KINDS:
        # Kept here for a clearer error if a future caller bypasses validation.
        raise ContractViolation(f"operation kind {kind!r} must run through Google Colab")

    import time

    started = time.monotonic()
    process = subprocess.Popen(
        args,
        cwd=None if cwd is None else os.fspath(cwd),
        env=merged_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=(os.name == "posix"),
        preexec_fn=_resource_preexec(effective, requested_seconds, requested_memory_gb),
    )
    try:
        stdout, stderr = process.communicate(timeout=requested_seconds)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_group(process)
        # Drain and close both pipes after the child exits.  This prevents
        # descriptor leaks in repeated guard tests and notebook retries.
        try:
            process.communicate(timeout=1.0)
        except subprocess.TimeoutExpired:
            _terminate_process_group(process)
            process.communicate()
        elapsed = time.monotonic() - started
        raise ContractViolation(
            f"guarded command exceeded {requested_seconds:.3f}s (elapsed {elapsed:.3f}s)"
        ) from exc
    elapsed = time.monotonic() - started
    result = GuardedCommandResult(
        argv=args,
        returncode=int(process.returncode),
        stdout=stdout,
        stderr=stderr,
        elapsed_seconds=elapsed,
        requested_memory_gb=float(requested_memory_gb),
        operation_kind=kind,
    )
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, args, output=stdout, stderr=stderr)
    return result


def validate_local_config(config: Mapping[str, object]) -> None:
    """Validate common config dictionaries before any local action."""

    execution = str(config.get("execution", config.get("runtime", "local"))).lower()
    operation = _normalise_kind(str(config.get("operation_kind", config.get("kind", "verify"))))
    if execution in {"colab", "google_colab", "google-colab"}:
        return
    if any(bool(config.get(name, False)) for name in ("training", "train", "sweep", "basin", "download", "use_gpu")):
        raise ContractViolation("local config requests training, a sweep, a download, or accelerator work")
    if operation in REMOTE_OPERATION_KINDS:
        raise ContractViolation(f"config operation {operation!r} requires Colab")
    if bool(config.get("use_gpu", config.get("gpu", False))):
        raise ContractViolation("local config requests GPU execution")
    validate_requested_resources(
        requested_cores=int(config.get("requested_cores", config.get("cores", config.get("max_cores", 2)))),
        requested_memory_gb=float(
            config.get("requested_memory_gb", config.get("memory_gb", config.get("max_ram_gb", 4.0)))
        ),
        requested_seconds=float(
            config.get("requested_seconds", config.get("timeout_seconds", config.get("max_seconds", 300.0)))
        ),
        operation_kind=operation,
    )


__all__ = [
    "ComputeBudget",
    "ContractViolation",
    "GuardedCommandResult",
    "LOCAL_OPERATION_KINDS",
    "REMOTE_OPERATION_KINDS",
    "THREAD_ENV_VARS",
    "assert_colab_execution",
    "assert_cpu_only",
    "infer_operation_kind",
    "is_colab_runtime",
    "local_context",
    "run_guarded",
    "validate_local_config",
    "validate_requested_resources",
]
